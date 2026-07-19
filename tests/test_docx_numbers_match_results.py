"""Thesis-vs-artifact guard — every headline number in the docx must trace to a result.

The 2026-06-27 campaign surfaced five failure modes where a number reported to the
reader had drifted from (or never existed in) the on-disk results:
  1. an orphan ``R2 0.884`` quoted from a retrospective ABM run whose artifact was
     gone (a number with no source);
  2. a "5x WIS 15-19" claim that presented a *static* baseline as if it were the
     *rolling* protocol (an over-claim);
  3. a stale model count (45 / 49 / 48 drifting across layers).

This guard reads the headline numbers straight out of the thesis docx and asserts
each one still equals the artifact it was reconciled against. If a future docx
edit (or a future re-run) makes them disagree, it fails loudly here instead of a
reviewer catching it. Only numbers with a verifiable on-disk source are checked;
anything without an artifact is intentionally left to manual review (we do not
invent a source just to assert against it).

Reads on-disk artifacts only (the docx + result JSON/CSV); no training, no DB,
no third-party deps (stdlib ``zipfile`` extracts the docx text — python-docx is
not required, keeping the guard portable).

Run (per-file, macOS): .venv/bin/python -m pytest tests/test_docx_numbers_match_results.py -q
"""

from __future__ import annotations

import csv
import json
import math
import re
import zipfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DOCX = _ROOT / "paper" / "고려대_보건석사학위논문_이승진_2026.docx"
_EVAL_CSV = _ROOT / "simulation" / "results" / "per_model_eval" / "per_model_metrics.csv"
_ABM_FWD = _ROOT / "simulation" / "results" / "abm_forward_validation" / "result.json"
_FIG1_SCRIPT = _ROOT / "simulation" / "scripts" / "regenerate_stale_thesis_figures.py"
_ABLATION = _ROOT / "simulation" / "results" / "abm_variant_ablation_anchored.json"
_FIGDATA = _ROOT / "simulation" / "results" / "abm_realism_figure_data.json"
_COUNTERFACTUAL = (
    _ROOT / "simulation" / "results" / "abm_counterfactual_ci" / "test.json"
)

_CHAMPION = "FusedEpi"


def _docx_text() -> str:
    """Flatten the thesis docx body to plain text (stripped of XML tags).

    Returns:
        The concatenated visible text of ``word/document.xml`` with markup
        removed. Numbers that render contiguously to the reader (e.g. ``3.28``)
        appear contiguously here because Word keeps a decimal token in one run.
    """
    if not _DOCX.exists():
        pytest.skip(f"{_DOCX} absent")
    with zipfile.ZipFile(_DOCX) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    text = re.sub(r"<[^>]+>", "", xml)
    return re.sub(r"&amp;", "&", text)


def _eval_rows() -> list[dict]:
    if not _EVAL_CSV.exists():
        pytest.skip(f"{_EVAL_CSV} absent (run the eval pipeline first)")
    return list(csv.DictReader(_EVAL_CSV.open(encoding="utf-8")))


def _champion_row() -> dict:
    for r in _eval_rows():
        if r.get("model") == _CHAMPION:
            return r
    pytest.skip(f"{_CHAMPION} absent from eval CSV")


def _abm_forward() -> dict:
    """The champion-anchored forward run whose R2 the thesis quotes.

    The artifact moved: the retired ``abm_multiorigin_forward/result.json`` nested the
    figures under ``base_origin_champion_anchored``; the live
    ``abm_forward_validation/result.json`` carries ``forward_r2`` and
    ``forward_r2_behavior_on/off`` at the top level. This guard silently *skipped* for
    as long as it pointed at the dead path — the exact silent-void failure it exists to
    prevent — so it now reads the live file and accepts either shape.
    """
    if not _ABM_FWD.exists():
        pytest.skip(f"{_ABM_FWD} absent")
    d = json.loads(_ABM_FWD.read_text(encoding="utf-8"))
    return d.get("base_origin_champion_anchored", d)


def _counterfactual() -> dict:
    """The headline vaccination counterfactual the thesis quotes in nine blocks.

    ``abm_counterfactual_ci`` used to be a single fixed output directory shared by every
    configuration, so a 20%-budget / 20-seed robustness sweep overwrote the headline run
    (30 seeds, 12,500 agents, 10% budget) that the thesis reports. The overwrite was
    invisible — the file still parsed and still had every key, it just held different
    numbers — and no guard compared it to the docx, so the thesis spent weeks citing
    figures no active artifact could reproduce. ``_paths()`` now routes sweeps elsewhere;
    this guard is the alarm that would have caught it either way.
    """
    if not _COUNTERFACTUAL.exists():
        pytest.skip(f"{_COUNTERFACTUAL} absent")
    d = json.loads(_COUNTERFACTUAL.read_text(encoding="utf-8"))
    meta = d.get("metadata", {})
    assert meta.get("n_seeds") == 30 and meta.get("n_agents") == 12_500, (
        f"active counterfactual artifact is NOT the headline config the thesis cites "
        f"(got n_seeds={meta.get('n_seeds')}, n_agents={meta.get('n_agents')}, "
        f"budget_frac={meta.get('budget_frac')}) — a sweep has overwritten it again; "
        f"re-run: .venv/bin/python -m simulation.scripts.abm_counterfactual_ci"
    )
    return d


def _assert_docx_has(text: str, token: str, artifact: float, label: str) -> None:
    """The rounded artifact value must appear verbatim in the docx text.

    Args:
        text: flattened docx body.
        token: the exact string the docx is expected to print (e.g. "3.28").
        artifact: the on-disk value the token was reconciled against.
        label: human name for the failure message.
    """
    assert token in text, (
        f"{label}: docx does not contain {token!r} (artifact={artifact}); "
        f"headline number drifted from the reported value or was re-rounded"
    )


def test_champion_wis_matches_eval() -> None:
    """Docx 'WIS 3.28' must equal the FusedEpi WIS in the eval CSV (rounded)."""
    row = _champion_row()
    wis = float(row["wis"])
    assert not math.isnan(wis)
    token = f"{wis:.2f}"  # 3.2784 -> "3.28"
    _assert_docx_has(_docx_text(), token, wis, "champion WIS")


def test_no_stale_rerank_wis() -> None:
    """The docx must NOT print the rerank cross-check WIS (4.263) as a champion number.

    ``rerank_champion.py`` recomputes a hold-out WIS on a different REAL_HORIZON
    slab (4.263) than the per_model_eval SSOT (3.28, the Table 1 value). A
    2026-06-27 champion-reframing edit mis-inserted 4.263 into the abstract while
    the tables kept 3.28 — an internal 3.28-vs-4.263 contradiction an external
    reviewer caught. The canonical champion test WIS is the per_model_eval value;
    the rerank cross-check value must never surface as a reported number. This is
    the failure the existing 'WIS 3.28 present' check could not catch (3.28 was
    still in the tables, so presence alone passed while the abstract disagreed).
    """
    assert "4.263" not in _docx_text(), (
        "docx contains '4.263' — the rerank cross-check WIS leaked back in; the "
        "canonical champion test WIS is the per_model_eval value (3.28), not the "
        "rerank's different-slab recomputation — reconcile abstract/body/tables"
    )


def test_champion_r2_matches_eval() -> None:
    """Docx 'R2 0.936' must equal the FusedEpi R2 in the eval CSV (rounded)."""
    row = _champion_row()
    r2 = float(row["r2"])
    assert not math.isnan(r2)
    token = f"{r2:.3f}"  # 0.9357 -> "0.936"
    _assert_docx_has(_docx_text(), token, r2, "champion R2")


def test_abm_forward_r2_matches_artifact() -> None:
    """Docx 'forward R2 0.722' must equal the ABM champion-anchored forward R2.

    This is the exact orphan/over-claim failure mode: the reported forward skill
    must come from ``base_origin_champion_anchored`` in the multi-origin result,
    not from a deleted retrospective run.
    """
    fwd = _abm_forward()
    val = float(fwd["forward_r2"])
    token = f"{val:.3f}"  # 0.72197... -> "0.722"
    _assert_docx_has(_docx_text(), token, val, "ABM forward R2")


def test_abm_behavior_on_off_match_artifact() -> None:
    """Docx behaviour-on/off forward R2 (0.557 / 0.041) must match the artifact.

    Guards the behaviour-mechanism claim: behaviour-on must read 0.557 and
    behaviour-off 0.041 exactly as stored, so the "behaviour is decisive" framing
    cannot drift from the numbers that justify it.
    """
    fwd = _abm_forward()
    text = _docx_text()
    on = float(fwd["forward_r2_behavior_on"])
    off = float(fwd["forward_r2_behavior_off"])
    _assert_docx_has(text, f"{on:.3f}", on, "ABM behaviour-on R2")
    _assert_docx_has(text, f"{off:.3f}", off, "ABM behaviour-off R2")


def test_model_count_matches_eval_rows() -> None:
    """Docx 'NN forecasting models' must equal the number of scored models.

    The eval CSV row count is the artifact the docx model count was reconciled
    against (G-379/2026-06-26). A stale 45/49 vs the live 48 would mean the
    reported lineup size no longer matches what was actually scored.
    """
    n = len(_eval_rows())
    text = _docx_text()
    assert re.search(rf"\b{n}\s+forecasting\s+models\b", text), (
        f"docx does not say '{n} forecasting models' but the eval CSV scored {n} "
        f"models — stale lineup count (re-reconcile docx ↔ per_model_metrics.csv)"
    )


def test_ablation_variant_r2_match_artifact() -> None:
    """§4.4 ablation R² (A 0.749 / B 0.681 / hybrid 0.795 / ensemble 0.753) must match
    the anchored ablation JSON. Added after the 2026-07-14 verify pass found the new
    §4.4/Appendix Q numbers were unguarded (a 'workplace roughly half' error slipped
    past the passing guard because nothing tied the ablation text to its artifact).
    """
    if not _ABLATION.exists():
        pytest.skip(f"{_ABLATION} absent")
    ab = json.loads(_ABLATION.read_text(encoding="utf-8"))
    v = ab["variants"]
    text = _docx_text()
    for key, label in (("A", "mean-field A"), ("B", "agent-to-agent B"),
                       ("H", "hybrid H")):
        val = float(v[key]["forward_r2"])
        _assert_docx_has(text, f"{val:.3f}", val, f"ablation {label} forward R2")
    ens = float(ab["ensemble_AB"]["forward_r2"])
    _assert_docx_has(text, f"{ens:.3f}", ens, "ablation A+B ensemble forward R2")


def test_ablation_layer_share_ordering_matches_figure() -> None:
    """§4.4 must say community is the LARGEST contact layer with workplace ~a third,
    matching the person-like layer_share in figure data — the reader compares this
    sentence directly to Figure Q.3, whose tallest bar is community. Guards against
    the reintroduction of the refuted 'workplace roughly half' claim.
    """
    if not _FIGDATA.exists():
        pytest.skip(f"{_FIGDATA} absent")
    fd = json.loads(_FIGDATA.read_text(encoding="utf-8"))
    ls = fd["person"]["layer_share"]
    top = max(ls, key=ls.get)
    assert top == "community", (
        f"figure data layer_share max is {top!r} (={ls[top]}), not community — "
        f"reconcile the person-like generation before asserting the docx text"
    )
    text = _docx_text()
    assert "workplace roughly half" not in text, (
        "docx still claims 'workplace roughly half' — refuted by the layer_share "
        f"data (community largest at {ls['community']:.2f}, workplace {ls['workplace']:.2f})"
    )
    _assert_docx_has(text, f"community the largest layer at {ls['community']:.2f}",
                     ls["community"], "§4.4 layer-share community")
    _assert_docx_has(text, f"workplace {ls['workplace']:.2f}",
                     ls["workplace"], "§4.4 layer-share workplace")


def test_fig1_metric_battery_count_matches_body() -> None:
    """Figure 1's '<N>-metric battery' caption must equal the body's metric count.

    The 2026-06-28 review caught a figure↔body drift every text-only check above
    is blind to: Figure 1 is an embedded PNG (``word/media/image1.png``), so its
    caption never reaches ``document.xml`` — this whole file read 6/6 green while
    the figure said '129-metric battery' and the body printed '124-metric' in four
    places. The 124→129 slip entered when the figure was regenerated to fix the
    retired-champion labels; nothing reconciled the figure's number back to the
    body.

    The figure caption is authored in ``regenerate_stale_thesis_figures.
    fig1_system_overview`` and a re-render reads it from there, so the script
    source is the figure's single source of truth. This guard pulls the
    ``<N>-metric battery`` count straight from that source and asserts it equals
    the count the body actually prints, so a future regeneration that re-introduces
    a mismatched number fails here instead of in a reviewer's hands. (OCR'ing the
    PNG would test it directly but needs a heavy dependency; guarding the source is
    portable and catches the same drift.)
    """
    if not _FIG1_SCRIPT.exists():
        pytest.skip(f"{_FIG1_SCRIPT} absent")
    src = _FIG1_SCRIPT.read_text(encoding="utf-8")
    fig_counts = {int(m) for m in re.findall(r"(\d+)-metric battery", src)}
    assert len(fig_counts) == 1, (
        f"figure-gen script prints inconsistent '<N>-metric battery' counts "
        f"{sorted(fig_counts)} — the Figure 1 caption must state one battery size"
    )
    fig_n = fig_counts.pop()
    body_counts = {int(m) for m in re.findall(r"(\d+)-metric", _docx_text())}
    assert body_counts, "docx body prints no '<N>-metric' count to reconcile against"
    assert body_counts == {fig_n}, (
        f"Figure 1 caption says {fig_n}-metric battery but the body prints "
        f"{sorted(body_counts)}-metric — figure↔body metric-count drift. The "
        f"figure is an embedded PNG, so the text-only guards miss it; re-render "
        f"fig1_system_overview to match the body (or reconcile the body) and "
        f"re-embed word/media/image1.png"
    )


def test_counterfactual_averted_per_dose_matches_artifact() -> None:
    """Docx '1.83 versus 1.38 infections averted per dose' must equal the headline run."""
    tests = _counterfactual()["tests"]["heterogeneous"]
    text = _docx_text()
    high = float(tests["high_contact_mean"])
    _assert_docx_has(text, f"{high:.2f}", high, "counterfactual high-contact averted/dose")
    # the uniform arm is not in tests{}; read it from the per-strategy CSV
    csv_path = _COUNTERFACTUAL.with_name("strategy_ci.csv")
    if not csv_path.exists():
        pytest.skip(f"{csv_path} absent")
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    uni = next(
        float(r["inf_averted_per_dose_mean"])
        for r in rows
        if r["population"] == "heterogeneous" and r["strategy"] == "uniform"
    )
    _assert_docx_has(text, f"{uni:.2f}", uni, "counterfactual uniform averted/dose")


def test_counterfactual_effect_size_and_seeds_match_artifact() -> None:
    """Cohen's dz, the null-control p-value, and the seed count must match the artifact."""
    d = _counterfactual()
    text = _docx_text()

    het = d["tests"]["heterogeneous"]
    dz = float(het["cohen_dz"])
    _assert_docx_has(text, f"{dz:.2f}", dz, "counterfactual Cohen's dz")
    assert float(het["paired_ttest_p"]) < 0.001, (
        "docx claims p < 0.001 for high-contact vs uniform, artifact says "
        f"{het['paired_ttest_p']}"
    )
    assert int(het["n_seed"]) == 30, f"docx says thirty seeds, artifact has {het['n_seed']}"

    # The homogeneous arm is the null control: no targeting signal when contact structure
    # is flat. If it ever goes significant, the headline claim loses its comparator.
    hom = d["tests"]["homogeneous"]
    assert float(hom["paired_ttest_p"]) > 0.05, (
        "homogeneous null control is no longer null "
        f"(p={hom['paired_ttest_p']}) — the thesis reports it as not significant"
    )
