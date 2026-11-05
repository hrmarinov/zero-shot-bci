"""Tests for the GPU health guard's parsing (the part that can silently disable the guard)."""

import math

import pytest

from src.gpu_health import parse_status


def test_parse_normal_line() -> None:
    s = parse_status("61, 2445 MHz, 20.13 W, 1834 MiB, Not Active, Not Active")
    assert (s.temp_c, s.sm_clock_mhz, s.power_w, s.mem_used_mb) == (61, 2445, 20.13, 1834)
    assert s.thermal_throttle is False


def test_parse_throttle_and_na_fields() -> None:
    s = parse_status("84, 1200 MHz, [N/A], 900 MiB, Active, Not Active")
    assert s.thermal_throttle is True and math.isnan(s.power_w)


def test_parse_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        parse_status("61, 2445 MHz")


def test_nan_temperature_does_not_stall(monkeypatch) -> None:
    import src.gpu_health as gh

    monkeypatch.setattr(gh, "free_ram_gb", lambda: 8.0)
    monkeypatch.setattr(gh, "query_gpu", lambda: parse_status("[N/A], 0 MHz, [N/A], 100 MiB, Not Active, Not Active"))
    monkeypatch.setattr(gh.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("guard paused")))
    assert math.isnan(gh.wait_until_healthy(log=lambda m: None).temp_c)
