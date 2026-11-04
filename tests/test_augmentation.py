"""Each augmentation must be seed-reproducible, shape-preserving, touch about half the trials,
and do what it claims (time reversal reverses, FT surrogate keeps the amplitude spectrum)."""

import numpy as np
import pytest
import torch

from src.adaptation.augmentation import AUGMENTATIONS, make_augmentation


def _batch() -> torch.Tensor:
    return torch.randn(200, 22, 1001, generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize("name", AUGMENTATIONS)
def test_reproducible_shape_preserving_and_partial(name: str) -> None:
    X = _batch()
    a = make_augmentation(name, seed=3)(X)
    b = make_augmentation(name, seed=3)(X)
    c = make_augmentation(name, seed=4)(X)
    assert a.shape == X.shape and torch.isfinite(a).all()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(a, c)
    changed = (~torch.isclose(a, X, atol=1e-6).all(dim=(1, 2))).float().mean().item()
    assert 0.35 < changed < 0.65  # probability 0.5 per trial


def test_time_reverse_reverses() -> None:
    X = _batch()
    out = make_augmentation("timereverse", seed=0)(X)
    flipped = torch.isclose(out, X.flip(-1)).all(dim=(1, 2))
    same = torch.isclose(out, X).all(dim=(1, 2))
    assert bool((flipped | same).all()) and flipped.any()


def test_ft_surrogate_keeps_amplitude_spectrum() -> None:
    X = _batch()
    out = make_augmentation("ftsurrogate", seed=0)(X)
    amp_in, amp_out = np.abs(np.fft.rfft(X.numpy(), axis=-1)), np.abs(np.fft.rfft(out.numpy(), axis=-1))
    # interior bins (DC/Nyquist handling may differ) keep their amplitude
    np.testing.assert_allclose(amp_out[..., 1:-1], amp_in[..., 1:-1], rtol=1e-3, atol=1e-3)


def test_unknown_name_rejected() -> None:
    with pytest.raises(ValueError):
        make_augmentation("mixup", seed=0)
