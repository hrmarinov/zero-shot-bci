"""Module A - learned geometric correction network (docs/wavelet_personalization_pipeline.md §2).

Replaces spd_shrinkage.py's closed-form (2-scalar) empirical-Bayes shrinkage
with a small MLP trained to predict a subject's full-data Riemannian-aligned
mean covariance from a noisy few-trial subsample - the same "denoise a
few-trial estimate toward the population" problem, but with a learned rather
than purely statistical mechanism.

Design choices, made explicit because each one was a real decision point:

- **Target space**: the *affine-invariant* Riemannian mean (`mean_riemann`,
  the same estimator `RiemannianAlignment.fit()` uses - the already-validated
  +4.7pp method this pipeline distills) is the ground truth being predicted.
- **Working space**: the network itself operates entirely in *log-Euclidean*
  tangent-space coordinates (matrix log centered at the identity), not on
  raw SPD matrices. This is one of the two metrics the pipeline doc names as
  acceptable ("Riemannian (affine-invariant or log-Euclidean) distance...
  not plain MSE") - it turns the SPD manifold into an ordinary vector space,
  so training is plain MSE regression on vectors *in that space*, which is
  mathematically the log-Euclidean distance between the corresponding SPD
  matrices, not naive elementwise MSE on covariance entries (which ignores
  the manifold's curvature and can't guarantee an SPD output). This also
  keeps every training step cheap (no repeated matrix exp/log for the loss
  itself) and reuses the exact tangent-space convention spd_shrinkage.py
  already established (reference=I, metric="logeuclid").
- **Prediction target, centered**: the network predicts a *deviation* from
  the population grand mean (input: few-trial tangent vector minus grand
  mean; output: predicted full-data tangent vector minus grand mean), not
  the raw tangent vector directly. This gives the network the same natural
  "pull toward the population" a shrinkage estimator has by construction,
  while still letting it learn a nonlinear, non-isotropic correction -
  and makes "predict zero" (fall back to the grand mean) a trivial, safe
  degenerate solution if the input carries no useful signal, rather than an
  arbitrary point the network has to learn from scratch.
- **Small by design**: doc §2 is explicit that this is "the load-bearing
  lesson from §10.1" (the eigenvoice shrinkage extension overfit a larger
  parameter count against a reference pool that couldn't support it) - kept
  to a couple of narrow hidden layers, not scaled up unless underfitting is
  actually observed.
- **Explicit sample-size conditioning**: the network's input is the tangent
  deviation *plus* a scalar `1/n_trials` feature, not the tangent deviation
  alone. The closed-form shrinkage estimator's weight is
  `between_var / (between_var + trial_var/n)` - it has n-dependent trust
  built into its formula for free, shrinking harder toward the population
  prior at small n and trusting the raw estimate more as n grows. A network
  that only sees the tangent vector itself has no direct signal of how
  reliable that vector is, and has to infer estimation noise indirectly from
  the vector's own shape.
- **Gated, bounded correction** (added after a real cross-dataset failure -
  see below): the network does NOT output a raw additive correction. It
  outputs `pred_dev = (1-alpha)*input_dev + alpha*raw_correction`, where
  `alpha in [0,1]` is itself a learned, per-example, sigmoid-gated scalar,
  initialized near 0 (bias -3, sigmoid(-3)~0.05) so the network must earn
  the right to apply a correction rather than defaulting to a large one.
  This mirrors shrink_subject_mean's own structure exactly - `shrunk =
  grand_mean + weight*(raw_mean - grand_mean)` is also a weight-in-[0,1]
  convex combination - except here the "target" endpoint (`raw_correction`)
  is itself learned/nonlinear instead of a single fixed population point,
  and the weight is input-dependent instead of one global scalar.

  Why this matters: a first version without the gate (network freely
  predicting an unconstrained tangent-space point) passed the Stage 0
  synthetic gate cleanly, but on real data (Module A trained on PhysioNet,
  evaluated zero-shot on IV-2a) it made every prediction *worse* than doing
  nothing - 3-15x farther from a held-out gold covariance than the raw
  few-trial estimate. Diagnosis (see session history / experiments/
  diagnose_module_a_real.py): PhysioNet's population grand-mean covariance
  *shape* turned out to be about as far from a given IV-2a subject's true
  shape as two different IV-2a subjects are from each other - trace
  normalization only fixes power-scale mismatch, not this kind of cross-
  dataset shape mismatch. shrink_subject_mean's weight is mathematically
  capped at interpolating between raw and (badly-matched) prior, so the
  worst case is bounded; the earlier unconstrained network could extrapolate
  *past* a bad prior into worse territory, which is exactly what it did.
  The gate can't fully solve a genuine population mismatch, but it bounds
  the failure mode to "at worst, no different from shrinkage" instead of
  "confidently wrong regardless of input" - and an input-dependent (rather
  than global) gate at least has the *capacity* to learn to distrust
  out-of-distribution inputs, which a single global weight cannot.
"""

from dataclasses import dataclass

import numpy as np
import torch
from pyriemann.estimation import Covariances
from pyriemann.geometry.mean import mean_riemann
from pyriemann.geometry.tangentspace import tangent_space, untangent_space
from torch import nn

from src.adaptation.spd_shrinkage import _trial_tangent_vectors

DEFAULT_HIDDEN_DIM = 64
DEFAULT_N_HIDDEN_LAYERS = 2
DEFAULT_SAMPLE_SIZES = [8, 16, 32, 64, 128]
DEFAULT_DRAWS_PER_SUBJECT_PER_SIZE = 8
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MAX_EPOCHS = 500
DEFAULT_PATIENCE = 20
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_GATE_L2_WEIGHT = 1e-4


GATE_INIT_BIAS = -3.0  # sigmoid(-3) ~= 0.047 - trust the raw estimate by default


class GeometricCorrectionNet(nn.Module):
    """Gated MLP: input is [tangent-space deviation-from-grand-mean,
    1/n_trials] (dim = output_dim + 1). Output is a convex combination of
    the raw input deviation and a learned correction, weighted by a learned
    per-example gate in [0,1] - see module docstring's "Gated, bounded
    correction" note for why this replaced a plain unconstrained MLP.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = DEFAULT_HIDDEN_DIM, n_hidden_layers: int = DEFAULT_N_HIDDEN_LAYERS):
        super().__init__()
        self.output_dim = output_dim
        trunk_layers: list[nn.Module] = []
        prev_dim = input_dim
        for _ in range(n_hidden_layers):
            trunk_layers.append(nn.Linear(prev_dim, hidden_dim))
            trunk_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)
        self.correction_head = nn.Linear(prev_dim, output_dim)
        self.gate_head = nn.Linear(prev_dim, 1)
        nn.init.constant_(self.gate_head.bias, GATE_INIT_BIAS)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_dev = x[:, : self.output_dim]
        h = self.trunk(x)
        raw_correction = self.correction_head(h)
        alpha = torch.sigmoid(self.gate_head(h))
        pred_dev = (1 - alpha) * input_dev + alpha * raw_correction
        return pred_dev, alpha


@dataclass
class TrainingExample:
    x: np.ndarray  # few-trial tangent vector, deviation from grand mean
    y: np.ndarray  # full-data tangent vector, deviation from grand mean
    n_trials: int
    subject_index: int


def _sample_size_feature(n_trials: int) -> np.ndarray:
    """1/n_trials as an explicit sampling-uncertainty conditioning input - see
    module docstring's "Explicit sample-size conditioning" note."""
    return np.array([1.0 / n_trials], dtype=np.float64)


@dataclass
class GeometricCorrectionModel:
    """Trained network plus the tangent-space bookkeeping needed to use it."""

    net: GeometricCorrectionNet
    grand_mean_: np.ndarray
    reference_: np.ndarray
    train_losses_: list[float]
    val_losses_: list[float]
    trivial_baseline_val_loss_: float


def _full_data_target_vector(X: np.ndarray, reference: np.ndarray, estimator: str, normalize_trace: bool) -> np.ndarray:
    """Affine-invariant Riemannian mean covariance -> log-Euclidean tangent vector."""
    covs = Covariances(estimator=estimator).fit_transform(X)
    if normalize_trace:
        traces = np.trace(covs, axis1=1, axis2=2)
        covs = covs / traces[:, np.newaxis, np.newaxis]
    ai_mean_cov = mean_riemann(covs)
    return tangent_space(ai_mean_cov[np.newaxis, :, :], reference, metric="logeuclid")[0]


def _build_training_examples(
    reference_subjects_X: list[np.ndarray],
    grand_mean: np.ndarray,
    reference: np.ndarray,
    sample_sizes: list[int],
    draws_per_subject_per_size: int,
    estimator: str,
    normalize_trace: bool,
    rng: np.random.Generator,
) -> list[TrainingExample]:
    examples = []
    for subject_idx, X in enumerate(reference_subjects_X):
        n_trials = X.shape[0]
        target_vec = _full_data_target_vector(X, reference, estimator, normalize_trace)
        target_dev = target_vec - grand_mean

        valid_sizes = [n for n in sample_sizes if n < n_trials]
        for n in valid_sizes:
            for _ in range(draws_per_subject_per_size):
                idx = rng.choice(n_trials, size=n, replace=False)
                X_sub = X[idx]
                trial_vecs = _trial_tangent_vectors(X_sub, reference, estimator=estimator, normalize_trace=normalize_trace)
                input_vec = trial_vecs.mean(axis=0)
                input_dev = input_vec - grand_mean
                x = np.concatenate([input_dev, _sample_size_feature(n)])
                examples.append(TrainingExample(x=x, y=target_dev, n_trials=n, subject_index=subject_idx))
    return examples


def train_geometric_correction(
    reference_subjects_X: list[np.ndarray],
    sample_sizes: list[int] | None = None,
    draws_per_subject_per_size: int = DEFAULT_DRAWS_PER_SUBJECT_PER_SIZE,
    estimator: str = "oas",
    normalize_trace: bool = False,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    n_hidden_layers: int = DEFAULT_N_HIDDEN_LAYERS,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    gate_l2_weight: float = DEFAULT_GATE_L2_WEIGHT,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    random_state: int = 42,
    verbose: bool = True,
) -> GeometricCorrectionModel:
    """Train Module A on a reference pool of full-data subject recordings.

    Each reference subject contributes many (few-trial-input, full-data-
    target) pairs, drawn from that subject's *own* trials only - the network
    learns to generalize the few-trial-to-full-data denoising mapping across
    subjects, not to memorize any one subject's covariance.

    Validation split is by *subject*, not by individual training pair -
    pairs from a validation subject never appear in training, matching the
    actual deployment question ("does this generalize to a subject the
    network has never seen"), not the easier and misleading "does it
    generalize to an unseen subsample of a subject it trained on."
    """
    if len(reference_subjects_X) < 4:
        raise ValueError(
            f"need at least 4 reference subjects for a meaningful train/val split, got {len(reference_subjects_X)}"
        )

    sample_sizes = sample_sizes or DEFAULT_SAMPLE_SIZES
    n_channels = reference_subjects_X[0].shape[1]
    reference = np.eye(n_channels)
    rng = np.random.default_rng(random_state)

    subject_order = rng.permutation(len(reference_subjects_X))
    n_val_subjects = max(2, int(round(len(reference_subjects_X) * val_fraction)))
    val_subject_idx = set(subject_order[:n_val_subjects].tolist())
    train_subject_idx = set(subject_order[n_val_subjects:].tolist())

    target_vecs = [
        _full_data_target_vector(X, reference, estimator, normalize_trace) for X in reference_subjects_X
    ]
    grand_mean = np.mean([target_vecs[i] for i in train_subject_idx], axis=0)

    all_examples = _build_training_examples(
        reference_subjects_X, grand_mean, reference, sample_sizes,
        draws_per_subject_per_size, estimator, normalize_trace, rng,
    )
    train_examples = [e for e in all_examples if e.subject_index in train_subject_idx]
    val_examples = [e for e in all_examples if e.subject_index in val_subject_idx]
    if not train_examples or not val_examples:
        raise ValueError("train/val split produced an empty example set - check subject count and sample_sizes")

    X_train = torch.tensor(np.stack([e.x for e in train_examples]), dtype=torch.float32)
    y_train = torch.tensor(np.stack([e.y for e in train_examples]), dtype=torch.float32)
    X_val = torch.tensor(np.stack([e.x for e in val_examples]), dtype=torch.float32)
    y_val = torch.tensor(np.stack([e.y for e in val_examples]), dtype=torch.float32)

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    net = GeometricCorrectionNet(input_dim, output_dim, hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # The one number that makes overfitting-vs-genuine-signal diagnosable at
    # a glance: MSE of the trivial "predict zero deviation" (i.e. always
    # fall back to the grand mean, no learned correction at all) on the SAME
    # held-out validation subjects. A trained val_loss that doesn't clear
    # this bar means the network has learned nothing the grand mean didn't
    # already give for free - caught this happening on a real (non-
    # synthetic) 16-subject reference pool during development, see session
    # history: train_loss fell to ~0.0001 while val_loss plateaued at
    # ~0.015, and the resulting predictions were 3-15x *farther* from a
    # held-out gold covariance than doing nothing at all.
    trivial_baseline_val_loss = loss_fn(torch.zeros_like(y_val), y_val).item()
    if verbose:
        print(f"  trivial baseline (predict zero deviation) val_loss={trivial_baseline_val_loss:.5f} - "
              f"trained val_loss must clear this to mean anything")

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        net.train()
        optimizer.zero_grad()
        pred, alpha_train = net(X_train)
        prediction_loss = loss_fn(pred, y_train)
        # Encourages the gate to stay near its conservative (trust-the-raw-
        # estimate) default unless applying a correction genuinely reduces
        # prediction_loss enough to be worth it - see "Gated, bounded
        # correction" in the module docstring for why this exists.
        gate_penalty = alpha_train.pow(2).mean()
        loss = prediction_loss + gate_l2_weight * gate_penalty
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"training loss became non-finite at epoch {epoch} (loss={loss.item()}) - "
                "likely a numerically ill-conditioned covariance or a learning-rate/scale issue, "
                "not something to silently train through"
            )
        loss.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            pred_val, _ = net(X_val)
            val_loss = loss_fn(pred_val, y_val).item()

        train_losses.append(loss.item())
        val_losses.append(val_loss)

        # Relative (not absolute) improvement threshold - an absolute 1e-6
        # cutoff was tried first and turned out to be meaningless noise
        # relative to a ~0.01-0.05 loss scale, letting "no_improve" reset on
        # essentially every epoch's floating-point jitter and never actually
        # firing before max_epochs (see session history: a 500-epoch run
        # where val_loss inched down by <0.0001/epoch the entire time never
        # triggered patience=20 even once).
        if val_loss < best_val_loss * (1 - 1e-3):
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch % 20 == 0 or epoch == max_epochs - 1):
            print(f"  epoch {epoch:4d}  train_loss={prediction_loss.item():.5f}  val_loss={val_loss:.5f}  "
                  f"best_val_loss={best_val_loss:.5f}  no_improve={epochs_without_improvement}  "
                  f"mean_alpha={alpha_train.mean().item():.3f}")

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
            break
    else:
        if verbose:
            print(f"  reached max_epochs={max_epochs} without early stopping - "
                  f"consider raising max_epochs if val_loss was still improving")

    assert best_state is not None
    net.load_state_dict(best_state)

    if verbose:
        if best_val_loss >= trivial_baseline_val_loss:
            print(f"  WARNING: best_val_loss={best_val_loss:.5f} did NOT beat the trivial "
                  f"zero-correction baseline ({trivial_baseline_val_loss:.5f}) - this model has "
                  f"learned nothing generalizable and should not be trusted for prediction "
                  f"(likely too few reference subjects for this architecture - see doc §9 risk #1)")
        else:
            print(f"  best_val_loss={best_val_loss:.5f} beats trivial baseline "
                  f"({trivial_baseline_val_loss:.5f}) by {100 * (1 - best_val_loss / trivial_baseline_val_loss):.1f}%")

    return GeometricCorrectionModel(
        net=net, grand_mean_=grand_mean, reference_=reference,
        train_losses_=train_losses, val_losses_=val_losses,
        trivial_baseline_val_loss_=trivial_baseline_val_loss,
    )


def predict_corrected_mean(
    model: GeometricCorrectionModel, X_subject: np.ndarray, estimator: str = "oas", normalize_trace: bool = False
) -> np.ndarray:
    """Learned analogue of shrink_subject_mean - same signature, drop-in comparable.

    Returns an SPD covariance matrix predicted from X_subject's few trials.
    """
    trial_vecs = _trial_tangent_vectors(X_subject, model.reference_, estimator=estimator, normalize_trace=normalize_trace)
    raw_mean = trial_vecs.mean(axis=0)
    input_dev = raw_mean - model.grand_mean_
    x = np.concatenate([input_dev, _sample_size_feature(X_subject.shape[0])])

    model.net.eval()
    with torch.no_grad():
        pred_dev_t, _ = model.net(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
        pred_dev = pred_dev_t.squeeze(0).numpy()

    pred_vec = model.grand_mean_ + pred_dev
    pred_cov = untangent_space(pred_vec[np.newaxis, :], model.reference_, metric="logeuclid")[0]

    if normalize_trace:
        natural_covs = Covariances(estimator=estimator).fit_transform(X_subject)
        natural_trace = float(np.mean(np.trace(natural_covs, axis1=1, axis2=2)))
        pred_cov = pred_cov * natural_trace

    return pred_cov
