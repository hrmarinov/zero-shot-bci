"""Prototypical-network meta-learning for cross-subject EEG personalization.

Structurally different from every population-based approach already tried
and found null-to-harmful in this project (spd_shrinkage.py's closed-form
shrinkage, geometric_correction.py's Module A) - see
docs/progress_and_direction.md's "External literature check" section for
the full comparison. Those methods use population data to produce a
*fixed* correction, applied zero-shot to a new subject/dataset; the
correction is baked in, so it fails badly when the target's covariance
shape doesn't resemble the reference population's (confirmed directly for
Module A: PhysioNet's population shape turned out to be about as far from
an IV-2a subject's true shape as two different IV-2a subjects are from
each other).

Here, population data (the *base* dataset) only trains an encoder to
produce a good discriminative *embedding space* - a transferable skill
("what makes same-class trials look similar"), not a specific answer. The
actual classification decision for a new subject is always a *prototype*
(mean embedding) computed fresh from that subject's own few labeled
trials, at deployment time, never baked in from the base population. This
can't fail the way Module A failed: the final decision never depends on
the target resembling the base population's shape, only on the learned
metric generalizing.

Two episode-sampling strategies are implemented:

- **Disjoint-subject episodes** (sample_episode, following Amirshahi et
  al.'s MetaWearS, Commun Med 2026, Algorithm 1): support and query
  subjects are non-overlapping per episode. Tried first, since it's the
  published design - but on real PhysioNet data, training didn't converge
  at all (loss stuck at exactly the uniform-guessing value for hundreds of
  episodes), even though the identical architecture/loss trained perfectly
  on synthetic data. Root-caused: this design forces the network to align
  *different subjects'* covariance structure inside every single episode's
  loss, and real cross-subject EEG heterogeneity is severe enough to
  prevent that from converging - see docs/progress_and_direction.md.
- **Same-subject episodes** (sample_same_subject_episode, the current
  default): support and query both come from *one* subject's own trials
  (a different, non-overlapping split) - directly matching the actual
  deployment task (IV-2a calibration -> eval is always the same subject,
  never a different one). This never requires aligning two different
  subjects within one episode's loss; cross-subject generalization still
  happens, but *across* episodes (each one samples a different random
  subject), never within one. Structurally sidesteps the exact problem
  that broke disjoint-subject training, and is a more honest match to
  what the real task is anyway.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from pyriemann.estimation import Covariances
from pyriemann.geometry.tangentspace import tangent_space
from torch import nn

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_HIDDEN_DIM = 64
DEFAULT_EMBEDDING_DIM = 32
DEFAULT_K_SHOT = 5
DEFAULT_M_QUERY = 5
DEFAULT_N_SUPPORT_SUBJECTS = 8
DEFAULT_N_QUERY_SUBJECTS = 8
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MAX_EPISODES = 3000
DEFAULT_PATIENCE_EVALS = 15
DEFAULT_EVAL_EVERY = 50
DEFAULT_VAL_FRACTION = 0.2


class ProtoEncoder(nn.Module):
    """Small MLP: log-Euclidean tangent vector of a single trial's own
    covariance -> embedding. Kept small deliberately - same "load-bearing
    lesson" as geometric_correction.py's Module A (docs/wavelet_
    personalization_pipeline.md §2): a reference pool of this scale can't
    support a large model without overfitting."""

    def __init__(self, input_dim: int, embedding_dim: int = DEFAULT_EMBEDDING_DIM, hidden_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def trial_tangent_vectors(X: np.ndarray, estimator: str = "oas", normalize_trace: bool = True) -> np.ndarray:
    """Per-trial covariances -> log-Euclidean tangent vectors (reference=I).

    normalize_trace defaults True here (unlike spd_shrinkage.py's default
    False) because this module's whole purpose is cross-dataset transfer
    (PhysioNet base -> IV-2a target, different recording hardware) - see
    spd_shrinkage.py's normalize_trace docstring for why this is provably
    harmless to the downstream classifier and necessary to remove a cross-
    hardware power-scale confound.
    """
    n_channels = X.shape[1]
    reference = np.eye(n_channels)
    covs = Covariances(estimator=estimator).fit_transform(X)
    if normalize_trace:
        traces = np.trace(covs, axis1=1, axis2=2)
        covs = covs / traces[:, np.newaxis, np.newaxis]
    return tangent_space(covs, reference, metric="logeuclid")


@dataclass
class SubjectTrials:
    """One episode-eligible subject's trials, grouped by class label."""

    X_by_class: dict[str, np.ndarray]  # label -> (n_trials, n_channels, n_times)


def _pooled_class_trials(subjects: list[SubjectTrials], subject_indices: np.ndarray, label: str) -> np.ndarray:
    arrays = [subjects[i].X_by_class[label] for i in subject_indices if label in subjects[i].X_by_class and len(subjects[i].X_by_class[label]) > 0]
    if not arrays:
        raise ValueError(f"no trials available for class {label!r} among the sampled episode subjects")
    return np.concatenate(arrays, axis=0)


def sample_episode(
    subjects: list[SubjectTrials],
    classes: list[str],
    n_support_subjects: int,
    n_query_subjects: int,
    k_shot: int,
    m_query: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """One episode: disjoint support/query subject pools, k_shot support +
    m_query query trials per class, pooled across each pool's subjects.
    """
    if len(subjects) < n_support_subjects + n_query_subjects:
        raise ValueError(
            f"need at least {n_support_subjects + n_query_subjects} subjects for disjoint support/query pools, "
            f"got {len(subjects)}"
        )
    order = rng.permutation(len(subjects))
    support_idx = order[:n_support_subjects]
    query_idx = order[n_support_subjects : n_support_subjects + n_query_subjects]

    support_X, support_y, query_X, query_y = [], [], [], []
    for label in classes:
        support_pool = _pooled_class_trials(subjects, support_idx, label)
        query_pool = _pooled_class_trials(subjects, query_idx, label)
        s_idx = rng.choice(len(support_pool), size=k_shot, replace=len(support_pool) < k_shot)
        q_idx = rng.choice(len(query_pool), size=m_query, replace=len(query_pool) < m_query)
        support_X.append(support_pool[s_idx])
        query_X.append(query_pool[q_idx])
        support_y += [label] * k_shot
        query_y += [label] * m_query

    return np.concatenate(support_X, axis=0), support_y, np.concatenate(query_X, axis=0), query_y


def sample_same_subject_episode(
    subjects: list[SubjectTrials],
    classes: list[str],
    k_shot: int,
    m_query: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """One episode: support and query both drawn from *one* randomly-chosen
    subject's own trials, a different non-overlapping split per class -
    matching the real deployment task exactly (calibration -> eval, always
    the same subject). Deliberately does NOT pool multiple subjects into
    one episode's prototype computation - doing so would silently
    reintroduce the cross-subject-alignment problem this sampling strategy
    exists to avoid (a shared prototype averaged across several subjects'
    trials is exactly the "align different subjects" computation that
    stalled disjoint-subject training). Call this multiple times per
    training step and average the resulting losses (see
    train_prototypical_network's episodes_per_step) for a smoother
    gradient without merging any subjects' trials together.
    """
    subject = subjects[rng.integers(len(subjects))]
    support_X, support_y, query_X, query_y = [], [], [], []
    for label in classes:
        trials = subject.X_by_class.get(label)
        if trials is None or len(trials) < 2:
            raise ValueError(f"subject has fewer than 2 trials for class {label!r} - cannot split into support/query")
        perm = rng.permutation(len(trials))
        n_support_here = min(k_shot, len(trials) - 1)
        n_query_here = min(m_query, len(trials) - n_support_here)
        support_X.append(trials[perm[:n_support_here]])
        query_X.append(trials[perm[n_support_here : n_support_here + n_query_here]])
        support_y += [label] * n_support_here
        query_y += [label] * n_query_here

    return np.concatenate(support_X, axis=0), support_y, np.concatenate(query_X, axis=0), query_y


def _prototypical_loss(
    encoder: ProtoEncoder, support_t: torch.Tensor, support_y_idx: torch.Tensor,
    query_t: torch.Tensor, query_y_idx: torch.Tensor, n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard Prototypical Networks loss (Snell et al. 2017): prototypes
    = per-class mean support embedding; query classified by softmax over
    negative squared Euclidean distance to each prototype."""
    support_emb = encoder(support_t)
    query_emb = encoder(query_t)

    embedding_dim = support_emb.shape[1]
    prototypes = torch.zeros(n_classes, embedding_dim, device=support_emb.device)
    for c in range(n_classes):
        prototypes[c] = support_emb[support_y_idx == c].mean(dim=0)

    dists = torch.cdist(query_emb, prototypes) ** 2  # (n_query, n_classes)
    log_probs = F.log_softmax(-dists, dim=1)
    loss = F.nll_loss(log_probs, query_y_idx)
    preds = log_probs.argmax(dim=1)
    acc = (preds == query_y_idx).float().mean()
    return loss, acc


def _episode_loss(
    encoder: ProtoEncoder, subjects: list[SubjectTrials], classes: list[str], class_to_idx: dict[str, int],
    n_classes: int, episode_mode: str, k_shot: int, m_query: int,
    n_support_subjects: int, n_query_subjects: int, rng: np.random.Generator, device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if episode_mode == "same_subject":
        X_s, y_s, X_q, y_q = sample_same_subject_episode(subjects, classes, k_shot, m_query, rng)
    elif episode_mode == "disjoint_subjects":
        X_s, y_s, X_q, y_q = sample_episode(subjects, classes, n_support_subjects, n_query_subjects, k_shot, m_query, rng)
    else:
        raise ValueError(f"unknown episode_mode={episode_mode!r}, expected 'same_subject' or 'disjoint_subjects'")

    support_t = torch.tensor(trial_tangent_vectors(X_s), dtype=torch.float32, device=device)
    query_t = torch.tensor(trial_tangent_vectors(X_q), dtype=torch.float32, device=device)
    support_y_idx = torch.tensor([class_to_idx[l] for l in y_s], device=device)
    query_y_idx = torch.tensor([class_to_idx[l] for l in y_q], device=device)
    return _prototypical_loss(encoder, support_t, support_y_idx, query_t, query_y_idx, n_classes)


def train_prototypical_network(
    train_subjects: list[SubjectTrials],
    classes: list[str],
    val_subjects: list[SubjectTrials] | None = None,
    episode_mode: str = "same_subject",
    episodes_per_step: int = 8,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    k_shot: int = DEFAULT_K_SHOT,
    m_query: int = DEFAULT_M_QUERY,
    n_support_subjects: int = DEFAULT_N_SUPPORT_SUBJECTS,
    n_query_subjects: int = DEFAULT_N_QUERY_SUBJECTS,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    max_episodes: int = DEFAULT_MAX_EPISODES,
    eval_every: int = DEFAULT_EVAL_EVERY,
    patience_evals: int = DEFAULT_PATIENCE_EVALS,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    device: torch.device | str = DEFAULT_DEVICE,
    random_state: int = 42,
    verbose: bool = True,
) -> ProtoEncoder:
    """Episodic meta-training. If val_subjects is None, train_subjects is
    split by *subject* (not by episode/trial) into train/val pools -
    subject-level splitting is required so validation genuinely measures
    generalization to unseen subjects, matching the actual deployment
    question, the same reasoning geometric_correction.py's train/val split
    already established for Module A.

    episode_mode="same_subject" (default, see module docstring for why) at
    episodes_per_step>1 draws that many *independent* single-subject
    episodes per gradient step and averages their losses - a smoother
    gradient than one subject per step, without ever pooling different
    subjects' trials into a shared prototype (each episode's prototypes
    are computed from one subject alone; only the resulting scalar losses
    are averaged).
    """
    rng = np.random.default_rng(random_state)
    torch.manual_seed(random_state)

    if val_subjects is None:
        order = rng.permutation(len(train_subjects))
        min_val = (n_support_subjects + n_query_subjects) if episode_mode == "disjoint_subjects" else 2
        n_val = max(min_val, int(round(len(train_subjects) * val_fraction)))
        val_subjects = [train_subjects[i] for i in order[:n_val]]
        train_subjects = [train_subjects[i] for i in order[n_val:]]

    input_dim = next(iter(train_subjects[0].X_by_class.values())).shape[1]
    input_dim = input_dim * (input_dim + 1) // 2  # tangent-space dim for n_channels

    encoder = ProtoEncoder(input_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    steps_per_episode = episodes_per_step if episode_mode == "same_subject" else 1

    best_val_acc = -1.0
    best_state = None
    evals_without_improvement = 0
    train_losses: list[float] = []

    for episode in range(max_episodes):
        encoder.train()
        optimizer.zero_grad()
        step_losses = []
        for _ in range(steps_per_episode):
            loss, _ = _episode_loss(
                encoder, train_subjects, classes, class_to_idx, n_classes, episode_mode,
                k_shot, m_query, n_support_subjects, n_query_subjects, rng, device,
            )
            step_losses.append(loss)
        loss = torch.stack(step_losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"episodic training loss became non-finite at episode {episode} (loss={loss.item()})")
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if (episode + 1) % eval_every == 0:
            encoder.eval()
            val_accs = []
            with torch.no_grad():
                for _ in range(10):
                    _, acc = _episode_loss(
                        encoder, val_subjects, classes, class_to_idx, n_classes, episode_mode,
                        k_shot, m_query, n_support_subjects, n_query_subjects, rng, device,
                    )
                    val_accs.append(acc.item())
            mean_val_acc = float(np.mean(val_accs))

            if verbose:
                recent_loss = float(np.mean(train_losses[-eval_every:]))
                print(f"  episode {episode + 1:5d}  train_loss={recent_loss:.4f}  "
                      f"val_episode_acc={mean_val_acc:.4f}  best={best_val_acc:.4f}")

            if mean_val_acc > best_val_acc + 1e-4:
                best_val_acc = mean_val_acc
                best_state = {k: v.clone() for k, v in encoder.state_dict().items()}
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
                if evals_without_improvement >= patience_evals:
                    if verbose:
                        print(f"  early stopping at episode {episode + 1} (no val improvement for "
                              f"{patience_evals} evals)")
                    break

    if best_state is not None:
        encoder.load_state_dict(best_state)
    return encoder


def compute_prototypes(encoder: ProtoEncoder, X_by_class: dict[str, np.ndarray], device: torch.device | str = DEFAULT_DEVICE) -> dict[str, np.ndarray]:
    """Prototype (mean embedding) per class from a target subject's own
    labeled trials - the only per-subject computation this mechanism does,
    always freshly derived from target data, never from the base
    population."""
    encoder.eval()
    prototypes = {}
    with torch.no_grad():
        for label, X in X_by_class.items():
            t = torch.tensor(trial_tangent_vectors(X), dtype=torch.float32, device=device)
            emb = encoder(t)
            prototypes[label] = emb.mean(dim=0).cpu().numpy()
    return prototypes


def predict_via_prototypes(
    encoder: ProtoEncoder, prototypes: dict[str, np.ndarray], X: np.ndarray, device: torch.device | str = DEFAULT_DEVICE,
) -> np.ndarray:
    """Classify trials by nearest prototype (negative squared Euclidean
    distance) in the learned embedding space."""
    encoder.eval()
    labels = list(prototypes.keys())
    proto_mat = torch.tensor(np.stack([prototypes[l] for l in labels]), dtype=torch.float32, device=device)
    with torch.no_grad():
        t = torch.tensor(trial_tangent_vectors(X), dtype=torch.float32, device=device)
        emb = encoder(t)
        dists = torch.cdist(emb, proto_mat) ** 2
        pred_idx = dists.argmin(dim=1).cpu().numpy()
    return np.array([labels[i] for i in pred_idx])
