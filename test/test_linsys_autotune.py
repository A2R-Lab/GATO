"""Host-tier gates for gato.linsys_autotune: fit / decide / persist / resolve.

No GPU, no solver modules — pure numpy logic on synthetic probe traces. The
GPU probe itself (tools/autotune_linsys.py) is timing-class and exercised in
quiet windows only.
"""
import numpy as np
import pytest

from gato.linsys_autotune import (decide_policy, fit_tau, load_tuning, lookup,
                                  resolve_linsys, save_tuning)

RNG = np.random.default_rng(3)


def synth_trace(n=600, cross=0.235, bdsv_ms=0.77, warm_ms=0.30, noise=0.0):
    """pred_err uniform [0, 0.4]; pcg cost linear in pred_err, crossing
    bdsv_ms exactly at `cross`."""
    pe = RNG.uniform(0.0, 0.4, n)
    slope = (bdsv_ms - warm_ms) / cross
    ms = warm_ms + slope * pe
    if noise:
        ms = ms + RNG.normal(0.0, noise, n)
    return pe, ms


def test_fit_tau_recovers_crossing():
    pe, ms = synth_trace()
    tau, diag = fit_tau(pe, ms, 0.77)
    assert tau == pytest.approx(0.235, abs=0.04)   # within ~a bin width
    assert diag["bdsv_ms"] == 0.77
    assert diag["bin_cost"] == sorted(diag["bin_cost"])  # monotone fit


def test_fit_tau_noisy_still_recovers():
    pe, ms = synth_trace(noise=0.03)
    tau, _ = fit_tau(pe, ms, 0.77)
    assert tau == pytest.approx(0.235, abs=0.06)


def test_fit_tau_pcg_always_wins():
    pe, ms = synth_trace()
    tau, _ = fit_tau(pe, ms, 10.0)                 # bdsv far above any pcg cost
    assert tau is None
    assert decide_policy(pe, tau) == {"policy": "pcg", "tau": None,
                                      "cold_frac": 0.0}


def test_fit_tau_bdsv_always_wins():
    pe, ms = synth_trace(warm_ms=1.0)              # even warm pcg above bdsv
    tau, _ = fit_tau(pe, ms, 0.77)
    assert tau == 0.0
    assert decide_policy(pe, tau)["policy"] == "bdsv"


def test_fit_tau_rejects_constant_trace():
    with pytest.raises(ValueError):
        fit_tau(np.full(100, 0.05), np.full(100, 0.3), 0.77)


def test_decide_policy_edges():
    pe = np.linspace(0.0, 0.4, 1000)
    d = decide_policy(pe, 0.2)                     # ~half cold -> auto
    assert d["policy"] == "auto" and d["tau"] == pytest.approx(0.2)
    assert decide_policy(pe, 0.399)["policy"] == "pcg"    # <2% cold
    assert decide_policy(pe, 0.001)["policy"] == "bdsv"   # >98% cold


def test_save_lookup_roundtrip(tmp_path):
    p = tmp_path / "tuning.json"
    entry = {"policy": "auto", "tau": 0.12, "provenance": {"date": "2026-08-12"}}
    save_tuning("iiwa14", 64, "fig8", entry, path=p)
    assert lookup("iiwa14", 64, "fig8", path=p) == entry
    assert lookup("iiwa14", 32, "fig8", path=p) is None
    assert lookup("iiwa14", 64, "other", path=p) is None
    # replace, and keep other keys
    save_tuning("iiwa14", 64, "fig8", {"policy": "pcg", "tau": None}, path=p)
    save_tuning("indy7", 64, "fig8", entry, path=p)
    assert lookup("iiwa14", 64, "fig8", path=p)["policy"] == "pcg"
    assert len(load_tuning(p)) == 2


def test_resolve_precedence(tmp_path):
    p = tmp_path / "tuning.json"
    save_tuning("indy7", 64, "fig8", {"policy": "bdsv_first", "tau": None}, path=p)
    save_tuning("indy7", 64, "kicky", {"policy": "auto", "tau": 0.31}, path=p)
    # explicit beats tuned
    assert resolve_linsys(False, "pcg", None, plant="indy7", N=64,
                          task_tag="fig8", path=p) == ("pcg", None)
    # tuned beats wired default
    assert resolve_linsys(False, None, None, plant="indy7", N=64,
                          task_tag="fig8", path=p) == ("bdsv_first", None)
    assert resolve_linsys(False, None, None, plant="indy7", N=64,
                          task_tag="kicky", path=p) == ("auto", 0.31)
    # explicit threshold beats the tuned tau
    assert resolve_linsys(False, None, 0.05, plant="indy7", N=64,
                          task_tag="kicky", path=p) == ("auto", 0.05)
    # unknown tag falls through to the wired default
    assert resolve_linsys(False, None, None, plant="indy7", N=64,
                          task_tag="nope", path=p) == ("auto", 0.1)


def test_resolve_wired_defaults():
    assert resolve_linsys(False) == ("auto", 0.1)          # fixed-base
    assert resolve_linsys(True) == ("bdsv", None)          # floating
    assert resolve_linsys(False, "auto", None) == ("auto", 0.1)
    assert resolve_linsys(False, "auto", 0.25) == ("auto", 0.25)
    assert resolve_linsys(True, "pcg", None) == ("pcg", None)
