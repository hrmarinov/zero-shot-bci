"""Every `src` module must import cleanly.

This is the guard that keeps the repository runnable: the experiment scripts are
expensive (they download datasets and train models), so a broken import in a
method module is otherwise only discovered after a long download. Importing is
cheap and needs no data.
"""

import importlib

import pytest

SRC_MODULES = [
    "src.config",
    "src.baselines",
    "src.datasets",
    "src.evaluate",
    "src.preprocessing",
    "src.synthetic_eeg",
    "src.adaptation.active_calibration",
    "src.adaptation.combat_align",
    "src.adaptation.coherent_point_drift",
    "src.adaptation.flow_field",
    "src.adaptation.geometric_correction",
    "src.adaptation.latency_alignment",
    "src.adaptation.prototypical_network",
    "src.adaptation.riemannian_align",
    "src.adaptation.riemannian_icp",
    "src.adaptation.spd_shrinkage",
    "src.adaptation.state_space_flow",
    "src.adaptation.wavelet_calibration",
]


@pytest.mark.parametrize("module_name", SRC_MODULES)
def test_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None
