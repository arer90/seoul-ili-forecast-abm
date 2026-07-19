#!/usr/bin/env python3
"""Promote §3.3/§3.4 prose symbols to inline OMML and cite the numbered equations.

Two defects the thesis carried:

1. Equations (3.1)-(3.24) are numbered by ``SEQ Equation`` fields, but no sentence
   anywhere in the document cited one — the numbers were decorative.
2. The surrounding prose typeset its symbols as *plain text* ("lambda_g", "beta(t)",
   "N_g", "R_t*(1-S/N)"), so the same quantity appeared in Cambria Math italic inside
   the display equations and as an ASCII underscore two lines later.

This script fixes both, in place, on the SSOT docx.

Line-height safety
------------------
The document's existing inline math (e.g. block 346's ``R₀≈β/γ``) is written in LINEAR
form — a slash run, never a stacked ``m:f`` fraction, never an ``m:nary`` sum. Stacked
constructs grow the line box and would reflow the page-locked layout. Every expression
built here therefore uses the same linear vocabulary: ``m:r`` for symbols, ``m:sSub`` /
``m:sSup`` for scripts, and upright ``m:sty="p"`` runs for operators and roman
abbreviations (VE, ifr, ref, min).

Run:
    .venv/bin/python scripts/thesis_s33_mathify.py            # apply
    .venv/bin/python scripts/check_page_lock.py               # MUST stay 369p / 31 anchors
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import docx
from lxml import etree

_ROOT = Path(__file__).resolve().parents[1]
_DOCX = _ROOT / "paper" / "고려대_보건석사학위논문_이승진_2026.docx"

_M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_XML = "http://www.w3.org/XML/1998/namespace"


def m(tag: str) -> str:
    return f"{{{_M}}}{tag}"


def w(tag: str) -> str:
    return f"{{{_W}}}{tag}"


# --- OMML construction (mirrors the document's own inline-math pattern) -------

def _cambria() -> etree._Element:
    rpr = etree.Element(w("rPr"))
    f = etree.SubElement(rpr, w("rFonts"))
    f.set(w("ascii"), "Cambria Math")
    f.set(w("hAnsi"), "Cambria Math")
    return rpr


def _mrun(text: str, upright: bool = False) -> etree._Element:
    """One math run. upright=True gives roman (operators, VE, ifr, ref, min)."""
    r = etree.Element(m("r"))
    if upright:
        mrpr = etree.SubElement(r, m("rPr"))
        etree.SubElement(mrpr, m("sty")).set(m("val"), "p")
    r.append(_cambria())
    t = etree.SubElement(r, m("t"))
    t.text = text
    if text != text.strip():
        t.set(f"{{{_XML}}}space", "preserve")
    return r


def _script(kind: str, base: tuple, scr: tuple) -> etree._Element:
    """m:sSub / m:sSup with the document's ctrlPr shape."""
    tag, pr_tag, slot = {
        "s": ("sSub", "sSubPr", "sub"),
        "S": ("sSup", "sSupPr", "sup"),
    }[kind]
    el = etree.Element(m(tag))
    pr = etree.SubElement(el, m(pr_tag))
    etree.SubElement(pr, m("ctrlPr")).append(_cambria())
    e = etree.SubElement(el, m("e"))
    e.append(_node(base))
    s = etree.SubElement(el, m(slot))
    s.append(_node(scr))
    return el


def _node(tok: tuple) -> etree._Element:
    kind = tok[0]
    if kind == "v":                      # italic variable
        return _mrun(tok[1])
    if kind == "r":                      # roman / upright (VE, ifr, operators)
        return _mrun(tok[1], upright=True)
    if kind in ("s", "S"):               # subscript / superscript
        return _script(kind, tok[1], tok[2])
    raise ValueError(f"bad token {tok!r}")


def omml(spec: list[tuple]) -> etree._Element:
    """Build an inline <m:oMath> from a token list."""
    om = etree.Element(m("oMath"))
    for tok in spec:
        om.append(_node(tok))
    return om


# --- paragraph surgery -------------------------------------------------------

def _runs(p) -> list:
    return [c for c in p.iterchildren() if c.tag == w("r")]


def replace(p, find: str, parts: list[tuple], *, occurrence: int = 0) -> None:
    """Replace ``find`` inside a w:r with a sequence of math/text parts.

    Args:
        find: exact substring, must exist in exactly one run (after prior edits).
        parts: [('m', spec) | ('t', literal)] — rendered left to right in place.
        occurrence: which matching run to use when several contain ``find``.

    Raises:
        LookupError: ``find`` not present (fail loud — a silent skip would leave the
            thesis half-converted, which is the exact failure mode this replaces).
    """
    hits = []
    for r in _runs(p):
        text = "".join(t.text or "" for t in r.iter(w("t")))
        if find in text:
            hits.append((r, text))
    if len(hits) <= occurrence:
        raise LookupError(f"not found (occurrence {occurrence}): {find!r}")
    r, text = hits[occurrence]

    head, _, tail = text.partition(find)
    rpr = r.find(w("rPr"))
    parent = r.getparent()
    idx = parent.index(r)

    def _wrun(s: str):
        nr = etree.Element(w("r"))
        if rpr is not None:
            nr.append(etree.fromstring(etree.tostring(rpr)))
        t = etree.SubElement(nr, w("t"))
        t.text = s
        t.set(f"{{{_XML}}}space", "preserve")
        return nr

    new: list = []
    if head:
        new.append(_wrun(head))
    for kind, payload in parts:
        new.append(omml(payload) if kind == "m" else _wrun(payload))
    if tail:
        new.append(_wrun(tail))

    for off, node in enumerate(new):
        parent.insert(idx + off, node)
    parent.remove(r)


def text_edit(p, old: str, new: str, *, occurrence: int = 0) -> None:
    """Plain text-for-text replacement inside one run (used for the equation citations)."""
    replace(p, old, [("t", new)], occurrence=occurrence)


# --- symbol vocabulary -------------------------------------------------------

V = lambda c: ("v", c)          # noqa: E731  italic variable
R = lambda s: ("r", s)          # noqa: E731  roman / operator
SUB = lambda b, s: ("s", b, s)  # noqa: E731
SUP = lambda b, s: ("S", b, s)  # noqa: E731

N_g = [SUB(V("N"), V("g"))]
N_h = [SUB(V("N"), V("h"))]
LAM_g = [SUB(V("λ"), V("g"))]
LAM_gt = [SUB(V("λ"), V("g")), R("("), V("t"), R(")")]
M_gh = [SUB(V("M"), V("gh"))]
C_gt = [SUB(V("c"), V("g")), R("("), V("t"), R(")")]
S_gt = [SUB(V("s"), V("g")), R("("), V("t"), R(")")]
BETA_t = [V("β"), R("("), V("t"), R(")")]
NU_g = [SUB(V("ν"), V("g"))]
OM_V = [SUB(V("ω"), V("V"))]
R_t = [SUB(V("R"), V("t"))]
D_g = [SUB(V("d"), V("g"))]


# --- the edit set ------------------------------------------------------------
# (block, kind, payload)  kind: 'math' -> replace(find, parts) | 'text' -> text_edit
EDITS: list[tuple[int, str, tuple]] = [
    # ---------- §3.3 block 339: skeleton lead-in ----------
    (339, "math", ("M_gh", [("m", M_gh)])),
    (339, "math", ("population N_g and force of infection lambda_g",
                   [("t", "population "), ("m", N_g),
                    ("t", " and force of infection "), ("m", LAM_g)])),
    (339, "text", ("the deterministic dynamics are:",
                   "the deterministic dynamics are Equations (3.1)–(3.6):")),

    # ---------- block 346: parameter glossary (Latin spellings -> Greek) ----------
    (346, "math", ("beta(t) is baseline transmissibility",
                   [("m", BETA_t), ("t", " is baseline transmissibility")])),
    (346, "math", ("sigma is the rate of progression",
                   [("m", [V("σ")]), ("t", " is the rate of progression")])),
    (346, "math", ("gamma is the recovery rate",
                   [("m", [V("γ")]), ("t", " is the recovery rate")])),
    (346, "math", ("delta is a nominal fatality-removal rate",
                   [("m", [V("δ")]), ("t", " is a nominal fatality-removal rate")])),
    (346, "math", ("nu_g is the district vaccination rate",
                   [("m", NU_g), ("t", " is the district vaccination rate")])),
    (346, "math", ("and omega is the slow waning",
                   [("t", "and "), ("m", [V("ω")]), ("t", " is the slow waning")])),
    # double hedge: the prose says "approximately" and the equation already carries ≈
    (346, "text", ("The basic reproduction number is approximately ",
                   "The basic reproduction number is ")),

    # ---------- block 347: behavioral layer ----------
    (347, "math", ("relaxes with time constant tau",
                   [("t", "relaxes with time constant "), ("m", [V("τ")])])),
    (347, "math", ("exceeds a personal threshold theta",
                   [("t", "exceeds a personal threshold "), ("m", [V("θ")])])),
    (347, "math", ("lowering c_g(t) and therefore",
                   [("t", "lowering "), ("m", C_gt), ("t", " and therefore")])),
    (347, "math", ("recovered compartment R_g of the",
                   [("t", "recovered compartment "), ("m", [SUB(V("R"), V("g"))]),
                    ("t", " of the")])),

    # ---------- block 348: implemented system ----------
    (348, "math", ("reduced force of infection (1−VE)λ_g and",
                   [("t", "reduced force of infection "),
                    ("m", [R("(1−"), R("VE"), R(")"), SUB(V("λ"), V("g"))]),
                    ("t", " and")])),
    (348, "math", ("at slow rates ω and ω_V, which",
                   [("t", "at slow rates "), ("m", [V("ω")]), ("t", " and "),
                    ("m", OM_V), ("t", ", which")])),
    (348, "math", ("recovery flow, γ·ifr·I_g, rather",
                   [("t", "recovery flow, "),
                    ("m", [V("γ"), R("·"), R("ifr"), R("·"), SUB(V("I"), V("g"))]),
                    ("t", ", rather")])),
    # citation FIRST: the math edit below splits this run at ", the full", which would
    # otherwise leave the citation's find string straddling two runs.
    (348, "text", ("the full per-district system integrated by the simulator is:",
                   "the full per-district system integrated by the simulator is given by "
                   "Equations (3.7)–(3.12):")),
    (348, "math", ("with district population N_g and force of infection λ_g,",
                   [("t", "with district population "), ("m", N_g),
                    ("t", " and force of infection "), ("m", LAM_g), ("t", ",")])),

    # ---------- block 355: symbol glossary ----------
    (355, "text", ("Every symbol in these equations has",
                   "Every symbol in Equations (3.1)–(3.14) has")),
    (355, "math", ("N_g is the resident population of district g and λ_g(t) its force",
                   [("m", N_g), ("t", " is the resident population of district "),
                    ("m", [V("g")]), ("t", " and "), ("m", LAM_gt), ("t", " its force")])),
    (355, "math", ("β(t) is the baseline transmission rate per day, σ the latent",
                   [("m", BETA_t), ("t", " is the baseline transmission rate per day, "),
                    ("m", [V("σ")]), ("t", " the latent")])),
    (355, "math", ("period), γ the recovery rate",
                   [("t", "period), "), ("m", [V("γ")]), ("t", " the recovery rate")])),
    (355, "math", ("period), ω the rate at which",
                   [("t", "period), "), ("m", [V("ω")]), ("t", " the rate at which")])),
    (355, "math", ("susceptible, ω_V the analogous",
                   [("t", "susceptible, "), ("m", OM_V), ("t", " the analogous")])),
    (355, "math", ("protection, ν_g the daily",
                   [("t", "protection, "), ("m", NU_g), ("t", " the daily")])),
    (355, "math", ("VE the leaky vaccine efficacy",
                   [("m", [R("VE")]), ("t", " the leaky vaccine efficacy")])),
    (355, "math", ("and ifr the infection-fatality ratio. M_gh is the entry",
                   [("t", "and "), ("m", [R("ifr")]),
                    ("t", " the infection-fatality ratio. "), ("m", M_gh),
                    ("t", " is the entry")])),
    (355, "math", ("and c_g(t) is the behavioral contact multiplier defined below",
                   [("t", "and "), ("m", C_gt),
                    ("t", " is the behavioral contact multiplier defined below")])),
    (355, "math", ("reproduction number is β/γ under homogeneous mixing, and the "
                   "time-varying reproduction number R_t is",
                   [("t", "reproduction number is "),
                    ("m", [V("β"), R("/"), V("γ")]),
                    ("t", " under homogeneous mixing, and the time-varying "
                          "reproduction number "), ("m", R_t), ("t", " is")])),
    (355, "math", ("not the within-district R_t alone",
                   [("t", "not the within-district "), ("m", R_t), ("t", " alone")])),
    (355, "math", ("the commuter matrix M and the district forces of infection λ_g define "
                   "a next-generation operator K whose spectral radius R* = ρ(K) governs",
                   [("t", "the commuter matrix "), ("m", [V("M")]),
                    ("t", " and the district forces of infection "), ("m", LAM_g),
                    ("t", " define a next-generation operator "), ("m", [V("K")]),
                    ("t", " whose spectral radius "),
                    ("m", [SUP(V("R"), R("∗")), R(" = "), V("ρ"), R("("), V("K"), R(")")]),
                    ("t", " governs")])),
    (355, "math", ("invades the network (R* > 1) or dies",
                   [("t", "invades the network ("),
                    ("m", [SUP(V("R"), R("∗")), R(" > "), R("1")]),
                    ("t", ") or dies")])),

    # ---------- block 356: force of infection ----------
    (356, "math", ("so λ_g(t) sums the prevalence",
                   [("t", "so "), ("m", LAM_gt), ("t", " sums the prevalence")])),
    (356, "text", ("weighted by how much time g’s residents spend there:",
                   "weighted by how much time g’s residents spend there, "
                   "as in Equation (3.13):")),

    # ---------- block 358: the two time-varying factors ----------
    (358, "math", ("The baseline transmissibility β(t) carries the optional seasonal "
                   "forcing β(t)=β₀·(1+ε·cos(2π(t−φ)/365.25)), whose amplitude ε is zero "
                   "by default so β stays constant",
                   [("t", "The baseline transmissibility "), ("m", BETA_t),
                    ("t", " carries the optional seasonal forcing "),
                    ("m", BETA_t + [R("="), SUB(V("β"), R("0")), R("·(1+"), V("ε"),
                                    R("·cos(2π("), V("t"), R("−"), V("φ"), R(")/365.25))")]),
                    ("t", ", whose amplitude "), ("m", [V("ε")]),
                    ("t", " is zero by default so "), ("m", [V("β")]),
                    ("t", " stays constant")])),
    (358, "math", ("The behavioral contact multiplier c_g(t)∈[0,1] is generated",
                   [("t", "The behavioral contact multiplier "),
                    ("m", C_gt + [R("∈[0,1]")]), ("t", " is generated")])),
    (358, "math", ("when a fraction s_g(t) of district",
                   [("t", "when a fraction "), ("m", S_gt), ("t", " of district")])),
    (358, "text", ("effective transmission is scaled down quadratically,",
                   "effective transmission is scaled down quadratically, "
                   "as in Equation (3.14):")),

    # ---------- block 360: cautions ----------
    (360, "math", ("every susceptible in g shares the same λ_g.",
                   [("t", "every susceptible in "), ("m", [V("g")]),
                    ("t", " shares the same "), ("m", LAM_g), ("t", ".")])),
    (360, "math", ("amplitude α and the fatigue weight κ to be",
                   [("t", "amplitude "), ("m", [V("α")]), ("t", " and the fatigue weight "),
                    ("m", [V("κ")]), ("t", " to be")])),
    (360, "math", ("whereas the threshold θ and fatigue time-constant τ trade",
                   [("t", "whereas the threshold "), ("m", [V("θ")]),
                    ("t", " and fatigue time-constant "), ("m", [V("τ")]), ("t", " trade")])),

    # ---------- block 365: epidemiological measures ----------
    (365, "math", ("computed as (R_g(T)+D_g(T))/N_g from",
                   [("t", "computed as "),
                    ("m", [R("("), SUB(V("R"), V("g")), R("("), V("T"), R(")+"),
                           SUB(V("D"), V("g")), R("("), V("T"), R("))/"),
                           SUB(V("N"), V("g"))]),
                    ("t", " from")])),
    (365, "math", ("basic reproduction number R₀=β/γ and, time-resolved",
                   [("t", "basic reproduction number "),
                    ("m", [SUB(V("R"), R("0")), R("="), V("β"), R("/"), V("γ")]),
                    ("t", " and, time-resolved")])),
    (365, "math", ("effective reproduction number R_t estimated",
                   [("t", "effective reproduction number "), ("m", R_t),
                    ("t", " estimated")])),

    # ---------- block 367: agent budget ----------
    (367, "math", ("living-population density d_g,",
                   [("t", "living-population density "), ("m", D_g), ("t", ",")])),

    # ---------- §3.4 block 378: the mechanistic anchor ----------
    (378, "math", ("the reproduction number R_t, the susceptible fraction S/N, and the "
                   "resulting force of infection R_t·(1−S/N)",
                   [("t", "the reproduction number "), ("m", R_t),
                    ("t", ", the susceptible fraction "),
                    ("m", [V("S"), R("/"), V("N")]),
                    ("t", ", and the resulting force of infection "),
                    ("m", R_t + [R("·(1−"), V("S"), R("/"), V("N"), R(")")])])),
    (378, "math", ("where n is the training length and n_ref a reference scale",
                   [("t", "where "), ("m", [V("n")]),
                    ("t", " is the training length and "),
                    ("m", [SUB(V("n"), R("ref"))]), ("t", " a reference scale")])),
    (378, "math", ("starts near the floor alpha_min on short records",
                   [("t", "starts near the floor "), ("m", [SUB(V("α"), R("min"))]),
                    ("t", " on short records")])),
]


def main() -> int:
    backup = _DOCX.with_name(_DOCX.stem + "_pre_mathify.docx")
    shutil.copy2(_DOCX, backup)

    d = docx.Document(str(_DOCX))
    kids = list(d.element.body.iterchildren())

    applied = 0
    for block, kind, payload in EDITS:
        p = kids[block]
        try:
            if kind == "math":
                find, parts = payload
                replace(p, find, parts)
            else:
                old, new = payload
                text_edit(p, old, new)
        except LookupError as exc:
            print(f"✗ block {block}: {exc}")
            print("  ABORT — nothing written; docx untouched.")
            return 1
        applied += 1

    d.save(str(_DOCX))
    print(f"✅ {applied}/{len(EDITS)} edits applied → {_DOCX.name}")
    print(f"   backup: {backup.name}")
    print("   next: .venv/bin/python scripts/check_page_lock.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
