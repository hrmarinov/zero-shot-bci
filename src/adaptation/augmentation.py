"""Training-time augmentation priors (braindecode 1.7.0 transforms), fixed a priori.

Settings come from Rommel et al. 2022 (IV-2a within-subject session 1 -> 2, the same
protocol), never tuned on the evaluation session: each transform is applied to a trial with
probability 0.5; FTSurrogate phase noise 0.9; FrequencyShift up to +-2.7 Hz; SmoothTimeMask
1.6 s. ChannelsDropout uses braindecode's default drop rate 0.2. SignFlip is left out because
ShallowConvNet's squaring layer makes it a no-op.
"""

from __future__ import annotations

from braindecode.augmentation import ChannelsDropout, FrequencyShift, FTSurrogate, SmoothTimeMask, TimeReverse
from torch import nn

PROBABILITY = 0.5
SFREQ = 250.0
AUGMENTATIONS = ("ftsurrogate", "freqshift", "timemask", "timereverse", "chandrop")


def make_augmentation(name: str, seed: int) -> nn.Module:
    """One braindecode transform with its own seeded random state (reproducible per seed)."""
    builders = {
        "ftsurrogate": lambda: FTSurrogate(PROBABILITY, phase_noise_magnitude=0.9, random_state=seed),
        "freqshift": lambda: FrequencyShift(PROBABILITY, sfreq=SFREQ, max_delta_freq=2.7, random_state=seed),
        "timemask": lambda: SmoothTimeMask(PROBABILITY, mask_len_samples=int(1.6 * SFREQ), random_state=seed),
        "timereverse": lambda: TimeReverse(PROBABILITY, random_state=seed),
        "chandrop": lambda: ChannelsDropout(PROBABILITY, p_drop=0.2, random_state=seed),
    }
    if name not in builders:
        raise ValueError(f"unknown augmentation {name!r}; choose from {AUGMENTATIONS}")
    return builders[name]()
