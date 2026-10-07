"""Phase 10 - state-space flow-field frequency features on real IV-2a data.

The state-space flow field mechanism (src/adaptation/state_space_flow.py)
is validated on synthetic ground truth: plane and frequency recovery
(stage0_state_space_flow_validation.py) and, most importantly, frequency's
unique reliability under a planted channel-mixing drift, where plane- and
covariance-based classifiers were unreliable (stage0_state_space_flow_eeg_validation.py).
None of this has touched real EEG until now - and that synthetic generator
*guarantees* clean rotational structure by construction, which real motor-
imagery EEG has no obligation to have. This is an honest first look at
whether real IV-2a trials carry any class-discriminative rotational-
frequency signal at all, not an assumption that they do.

Per subject: for every trial (calibration and eval), fit a fresh *local*
PCA (2 components) and a flow field on that trial's own 22-channel, 501-
sample window alone - no calibration-fixed projection, exactly the
mechanism validated as drift-robust. The plain (non-debiased) fit is used
here, not the debiased one: fit_skew_symmetric_flow_debiased needs a known
noise variance, which the synthetic validation could supply exactly but
real data cannot without a real estimation step (not built here). This
matters less for a *nearest-template* classifier than it would for reading
off an absolute frequency value, since the same attenuation bias applies
consistently across all trials - a caveat carried forward, not resolved.

Per-class template frequency = mean of that class's calibration trials'
own recovered frequencies (4 classes, unlike the synthetic 2-class
validation). Eval trials classified by nearest template frequency alone -
the specific mechanism found uniquely drift-robust, tested first before
anything more elaborate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.decomposition import PCA

from src.adaptation.state_space_flow import dominant_rotation, fit_skew_symmetric_flow
from src.config import IV2A_CLASSES, IV2A_SUBJECTS, RESULTS_DIR
from src.datasets import load_iv2a_subject
from src.evaluate import accuracy_summary, wilson_interval

DT = 1.0


def trial_frequency(X_trial: np.ndarray) -> float:
    """X_trial: (n_channels, n_times) - this project's standard trial
    layout. Transposed to (n_times, n_channels) for PCA/flow fitting,
    matching stage0_state_space_flow_eeg_validation.py's convention."""
    Y = X_trial.T
    pca = PCA(n_components=2)
    Z = pca.fit_transform(Y)
    M_skew, _, _ = fit_skew_symmetric_flow(Z, dt=DT)
    omega_fit, _ = dominant_rotation(M_skew, Z)
    return omega_fit


def run() -> pd.DataFrame:
    rows = []
    for subject in IV2A_SUBJECTS:
        print(f"[subject {subject}] loading...")
        data = load_iv2a_subject(subject)

        print(f"[subject {subject}] extracting per-trial frequencies "
              f"({len(data.X_calib)} calib + {len(data.X_eval)} eval trials)...")
        freq_calib = np.array([trial_frequency(data.X_calib[i]) for i in range(len(data.X_calib))])
        freq_eval = np.array([trial_frequency(data.X_eval[i]) for i in range(len(data.X_eval))])

        template = {c: float(freq_calib[data.y_calib == c].mean()) for c in IV2A_CLASSES}
        y_pred = np.array([min(template, key=lambda c: abs(template[c] - f)) for f in freq_eval])

        acc, n_correct, n_total = accuracy_summary(data.y_eval, y_pred)
        ci_low, ci_high = wilson_interval(n_correct, n_total)
        p_vs_chance = binomtest(n_correct, n_total, p=0.25, alternative="greater").pvalue

        print(f"[subject {subject}] template omegas={ {c: round(v, 3) for c, v in template.items()} }  "
              f"acc={acc:.3f} [{ci_low:.3f}, {ci_high:.3f}]  p_vs_chance={p_vs_chance:.4f}")
        rows.append({
            "subject": subject, "accuracy": acc, "ci_low": ci_low, "ci_high": ci_high,
            "n_correct": n_correct, "n_total": n_total, "p_vs_chance": p_vs_chance,
            **{f"template_omega_{c}": template[c] for c in IV2A_CLASSES},
        })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "phase10_state_space_flow_frequency.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")
    print(f"\nMean accuracy: {df['accuracy'].mean():.3f} (std {df['accuracy'].std():.3f}), chance=0.250")
    print(df[["subject", "accuracy", "p_vs_chance"]].to_string(index=False))
    return df


if __name__ == "__main__":
    run()
