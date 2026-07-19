"""G-310 (2026-06-18): SHAP must explain ONE champion per active model, not every *.pt.

Root cause: champion-challenger writes versioned .pt (<name>.pt / <name>_deploy.pt /
<name>_attempt_vN_<ts>.pt) and clean() did not archive models/, so models/ accumulated
cross-run + retired artifacts (231). SHAP globbed *.pt → explained all 231 (4-5x waste +
stale pollution). _select_champion_pts filters to the active lineup's finals (one each).

macOS: run PER-FILE.
"""
import os
from pathlib import Path

from simulation.pipeline.shap_analysis import _select_champion_pts, _active_model_names


def _touch(p: Path, mtime: float | None = None):
    p.write_bytes(b"x")
    if mtime is not None:
        os.utime(p, (mtime, mtime))


def test_g310_one_per_active_prefers_plain(tmp_path):
    """A model with BOTH plain and versioned .pt → the plain <name>.pt (eval champion) wins."""
    _touch(tmp_path / "GAT.pt")
    _touch(tmp_path / "GAT_deploy.pt")
    _touch(tmp_path / "GAT_attempt_v2_20260613_101010.pt")
    sel = _select_champion_pts(tmp_path, ["GAT"])
    assert sel == [("GAT", tmp_path / "GAT.pt")], "plain <name>.pt preferred over deploy/attempt"


def test_g310_fallback_to_newest_when_no_plain(tmp_path):
    """No plain <name>.pt → newest <name>_*.pt by mtime (the champion artifact survives)."""
    _touch(tmp_path / "SVR-RBF_attempt_v2_20260613_010101.pt", mtime=1_000_000)
    _touch(tmp_path / "SVR-RBF_attempt_v4_20260618_032503.pt", mtime=2_000_000)  # newer
    sel = _select_champion_pts(tmp_path, ["SVR-RBF"])
    assert sel == [("SVR-RBF", tmp_path / "SVR-RBF_attempt_v4_20260618_032503.pt")]


def test_g310_excludes_retired_and_crossrun(tmp_path):
    """Retired / non-active models on disk are NOT explained (the 231→~53 fix)."""
    _touch(tmp_path / "SVR-RBF.pt")           # active
    _touch(tmp_path / "CatBoost.pt")          # retired (not in active list)
    _touch(tmp_path / "Chronos-2-FT-Real.pt")  # retired
    _touch(tmp_path / "TSIR_attempt_v2_20260612_220000.pt")  # retired cross-run
    sel = _select_champion_pts(tmp_path, ["SVR-RBF", "ElasticNet"])
    names = [n for n, _ in sel]
    assert names == ["SVR-RBF"], "only active models with an artifact; retired excluded"


def test_g310_skips_active_without_artifact(tmp_path):
    """An active model that FAILED this run (no .pt) is skipped, not errored."""
    _touch(tmp_path / "SVR-RBF.pt")
    sel = _select_champion_pts(tmp_path, ["SVR-RBF", "TabPFN", "OverseasTransfer"])
    assert [n for n, _ in sel] == ["SVR-RBF"], "TabPFN/OverseasTransfer (no .pt) skipped"


def test_g310_no_name_prefix_bleed(tmp_path):
    """'DNN' must not capture 'DNN-Conformal' artifacts (underscore boundary)."""
    _touch(tmp_path / "DNN_attempt_v3_20260617_191529.pt")
    _touch(tmp_path / "DNN-Conformal.pt")
    sel = _select_champion_pts(tmp_path, ["DNN", "DNN-Conformal"])
    d = dict(sel)
    assert d["DNN"] == tmp_path / "DNN_attempt_v3_20260617_191529.pt"
    assert d["DNN-Conformal"] == tmp_path / "DNN-Conformal.pt"


def test_g310_missing_dir_returns_empty(tmp_path):
    assert _select_champion_pts(tmp_path / "nope", ["SVR-RBF"]) == []


def test_g310_active_names_is_53_and_has_champion():
    """SSOT flatten: CATEGORY_MODELS → ~53 active names incl. the champion family."""
    names = _active_model_names()
    assert "SVR-RBF" in names and "ElasticNet" in names
    assert 40 <= len(names) <= 70, f"active lineup sanity (got {len(names)})"
    assert len(names) == len(set(names)), "no duplicate model names"
