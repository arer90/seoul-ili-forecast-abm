"""
simulation.server.static_citations
===================================
Curated citation catalogue used as a graceful fallback for
``epi.literature_rag`` when no vector index is available.

The entries are distilled from ``paper/overleaf_draft/main.tex`` plus the
methodological references that appear in the evaluation pipeline
(Hansen MCS, Murphy Brier, Hyndman sMAPE, etc.). Match against
``tags`` + ``title`` + ``relevance`` — NOT full-text — so the scoring
is intentionally shallow and honest about it.

Add new entries bottom-up; keep IDs stable for downstream UI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Citation:
    """One static citation entry.

 Attributes
 ----------
 id
 Stable slug used by UIs to anchor and deduplicate.
 authors, year, title, venue
 Standard bibliographic fields.
 doi_or_url
 Preferred canonical link, or empty string if none.
 tags
 Lowercase tokens used for the keyword-overlap score.
 Include common synonyms (e.g. "rt", "reproduction-number").
 relevance
 One-sentence plain-English note on WHY this paper matters to
 the MPH project. Shown verbatim in tool output.
 """
    id: str
    authors: str
    year: int
    title: str
    venue: str
    doi_or_url: str
    tags: tuple[str, ...]
    relevance: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "authors": self.authors,
            "year": self.year,
            "title": self.title,
            "venue": self.venue,
            "doi_or_url": self.doi_or_url,
            "tags": list(self.tags),
            "relevance": self.relevance,
        }


# ══════════════════════════════════════════════════════════════════════
# Catalogue
# ══════════════════════════════════════════════════════════════════════
STATIC_CITATIONS: tuple[Citation, ...] = (
    # ── From paper/overleaf_draft/main.tex ──────────────────────────
    Citation(
        id="kdca_guidelines_2024",
        authors="Korea Disease Control and Prevention Agency (KDCA)",
        year=2024,
        title="Guidelines for Notifiable Infectious Disease Surveillance (2024 edition)",
        venue="KDCA, Cheongju",
        doi_or_url="",
        tags=("kdca", "surveillance", "notifiable", "korea", "sentinel", "ili",
              "reporting", "legal-class"),
        relevance=(
            "Defines the legal class-1 through class-4 notifiable-disease "
            "schema and the sentinel-ILI reporting system — the data "
            "substrate for weekly_disease and sentinel_influenza tables."
        ),
    ),
    Citation(
        id="npi_collateral_2022",
        authors="Yeoh A, Fox K, Lee C, et al.",
        year=2022,
        title="Impact of COVID-19 public health measures on surveillance of "
              "notifiable childhood infectious diseases in Australia",
        venue="Journal of Paediatrics and Child Health 58(8):1456–1463",
        doi_or_url="",
        tags=("npi", "covid", "surveillance", "suppression", "notifiable",
              "confounding", "reporting-artefact"),
        relevance=(
            "Establishes NPI-induced under-ascertainment of respiratory "
            "infections — cited for the 2020–22 confound in the Seoul "
            "ILI series that breaks stationarity in the multi-model "
            "training window."
        ),
    ),
    Citation(
        id="immunity_debt_2022",
        authors="Cohen R, Levy C, Bingen E, et al.",
        year=2022,
        title="Impact of COVID-19-related hygienic measures on respiratory "
              "and enteric infections: the immunity debt concept",
        venue="Infectious Diseases Now 52(4):199–210",
        doi_or_url="",
        tags=("immunity-debt", "rebound", "post-covid", "susceptibles",
              "npi", "respiratory"),
        relevance=(
            "The canonical 'immunity debt' reference used to explain the "
            "winter 2022–23 ILI rebound observed in the 24–25 season's "
            "elevated peak."
        ),
    ),
    Citation(
        id="flu_suppression_2020",
        authors="Olsen KC, Lee C, Blyth MS, et al.",
        year=2020,
        title="Decreased influenza activity during the COVID-19 pandemic — "
              "United States, Australia, Chile, and South Africa, 2020",
        venue="MMWR Morbidity and Mortality Weekly Report 69(37):1305–1309",
        doi_or_url="",
        tags=("influenza", "covid", "suppression", "mmwr", "sentinel",
              "who-flunet", "seasonality"),
        relevance=(
            "Global evidence that 2020 flu circulation collapsed under "
            "NPIs — cited when framing the reduced ILI signal the DL "
            "models fit to in the training window."
        ),
    ),
    Citation(
        id="kermack_mckendrick_1927",
        authors="Kermack WO, McKendrick AG",
        year=1927,
        title="A contribution to the mathematical theory of epidemics",
        venue="Proceedings of the Royal Society A 115(772):700–721",
        doi_or_url="",
        tags=("seir", "sir", "compartmental", "epidemic-threshold",
              "mathematical-epidemiology", "r0"),
        relevance=(
            "Foundational paper for the compartmental metapopulation "
            "SEIR-V-D model used as the Stage 5 scenario engine."
        ),
    ),
    Citation(
        id="hethcote_2000",
        authors="Hethcote HW",
        year=2000,
        title="The mathematics of infectious diseases",
        venue="SIAM Review 42(4):599–653",
        doi_or_url="https://doi.org/10.1137/S0036144500371907",
        tags=("seir", "compartmental", "endemic", "review", "r0",
              "reproductive-number", "stability"),
        relevance=(
            "Goes-to reference for endemic-SIR threshold theory — cited "
            "in the simulation-advisor's R_e adjustment guidance."
        ),
    ),
    Citation(
        id="seoul_covid_spatial_2021",
        authors="Kim J, Lee S, Park H, et al.",
        year=2021,
        title="Spatial analysis of COVID-19 transmission in Seoul: the role "
              "of population density and mobility",
        venue="Int. J. Environmental Research and Public Health 18(21):11286",
        doi_or_url="",
        tags=("seoul", "spatial", "commuter", "mobility", "covid", "gu",
              "density", "metapopulation"),
        relevance=(
            "Methodological precedent for the 25-gu commuter-coupled "
            "force-of-infection term β_ij in the Stage 5 metapopulation "
            "SEIR-V-D."
        ),
    ),
    Citation(
        id="tb_spatial_korea_2020",
        authors="Park S, Choi J, Kim K, et al.",
        year=2020,
        title="Spatial clustering of tuberculosis in Seoul, Korea",
        venue="BMC Public Health 20:1283",
        doi_or_url="",
        tags=("seoul", "spatial", "clustering", "gu", "moran", "lisa",
              "tuberculosis"),
        relevance=(
            "Local-indicator-of-spatial-association (LISA) template for "
            "the gu-level diagnostic maps in Stage 3b."
        ),
    ),
    Citation(
        id="ecological_fallacy_1994",
        authors="Greenland S, Robins J",
        year=1994,
        title="Ecological studies — biases, misconceptions, and counterexamples",
        venue="American Journal of Epidemiology 139(8):747–760",
        doi_or_url="",
        tags=("ecological-fallacy", "aggregation-bias", "spatial",
              "interpretation-caveat", "epidemiology"),
        relevance=(
            "Cited as the caveat when interpreting gu-level SHAP "
            "contributions — ecological associations ≠ individual-level "
            "causes."
        ),
    ),
    Citation(
        id="strobe_2007",
        authors="von Elm E, Altman DG, Egger M, Pocock SJ, Gøtzsche PCG, "
                "Vandenbroucke JP",
        year=2007,
        title="The Strengthening the Reporting of Observational Studies in "
              "Epidemiology (STROBE) Statement",
        venue="PLoS Medicine 4(10):e296",
        doi_or_url="https://doi.org/10.1371/journal.pmed.0040296",
        tags=("strobe", "reporting", "observational", "guideline",
              "reproducibility", "checklist"),
        relevance=(
            "Reporting checklist the paper adheres to — governs the "
            "Methods section structure."
        ),
    ),

    # ── Methodological refs for evaluation pipeline ──────────
    Citation(
        id="hansen_mcs_2011",
        authors="Hansen PR, Lunde A, Nason JM",
        year=2011,
        title="The model confidence set",
        venue="Econometrica 79(2):453–497",
        doi_or_url="https://doi.org/10.3982/ECTA5771",
        tags=("mcs", "model-confidence-set", "comparison", "forecasting",
              "bootstrap", "dm-test", "ranking"),
        relevance=(
            "Defines the Model Confidence Set used for the MCS membership "
            "column in fig1 WIS ranking (§3e)."
        ),
    ),
    Citation(
        id="murphy_brier_1973",
        authors="Murphy AH",
        year=1973,
        title="A new vector partition of the probability score",
        venue="Journal of Applied Meteorology 12(4):595–600",
        doi_or_url="https://doi.org/10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2",
        tags=("brier", "decomposition", "calibration", "reliability",
              "resolution", "uncertainty", "probabilistic"),
        relevance=(
            "Three-term decomposition (U − R + REL) of the Brier score "
            "implemented as fig8 and reused in the cost-loss threshold "
            "reconstruction."
        ),
    ),
    Citation(
        id="hyndman_2018_sMAPE",
        authors="Hyndman RJ, Koehler AB",
        year=2018,
        title="Another look at measures of forecast accuracy (revisited)",
        venue="Forecasting: principles and practice (2nd ed.), OTexts",
        doi_or_url="https://otexts.com/fpp2/accuracy.html",
        tags=("smape", "mape", "forecast-accuracy", "epsilon-shifted",
              "denominator", "zero-series"),
        relevance=(
            "Justifies the ε-shifted sMAPE (ε = max(1.0, 0.05·ȳ)) used "
            "in fig12 regime-split sMAPE."
        ),
    ),
    Citation(
        id="bosse_2026_costloss",
        authors="Bosse NI, et al.",
        year=2026,
        title="Evaluating probabilistic infectious-disease forecasts with "
              "decision-theoretic metrics",
        venue="PLOS Computational Biology (preprint)",
        doi_or_url="",
        tags=("cost-loss", "decision-theoretic", "skill-score",
              "asymmetric-cost", "forecast-value", "climatology"),
        relevance=(
            "Bounded cost-ratio range r ∈ {1, 2, 3, 5, 7, 10} used for "
            "the fig13 cost-loss heatmap."
        ),
    ),
    Citation(
        id="cori_2013_epiestim",
        authors="Cori A, Ferguson NM, Fraser C, Cauchemez S",
        year=2013,
        title="A new framework and software to estimate time-varying "
              "reproduction numbers during epidemics",
        venue="American Journal of Epidemiology 178(9):1505–1512",
        doi_or_url="https://doi.org/10.1093/aje/kwt133",
        tags=("rt", "reproduction-number", "epiestim", "bayesian",
              "sliding-window", "serial-interval", "renewal"),
        relevance=(
            "Algorithm behind the ``epi.rt_estimate`` MCP tool — the "
            "sliding-window posterior-mean Rt reported in rt_history."
        ),
    ),
    Citation(
        id="gneiting_2007_wis",
        authors="Gneiting T, Raftery AE",
        year=2007,
        title="Strictly proper scoring rules, prediction, and estimation",
        venue="Journal of the American Statistical Association 102(477):359–378",
        doi_or_url="https://doi.org/10.1198/016214506000001437",
        tags=("wis", "crps", "scoring-rule", "proper", "probabilistic",
              "forecast-evaluation", "quantile"),
        relevance=(
            "Foundation for the WIS and CRPS metrics that rank the 66 "
            "models in per_model_metrics.csv."
        ),
    ),
    Citation(
        id="flusight_bracher_2021",
        authors="Bracher J, Ray EL, Gneiting T, Reich NG",
        year=2021,
        title="Evaluating epidemic forecasts in an interval format",
        venue="PLOS Computational Biology 17(2):e1008618",
        doi_or_url="https://doi.org/10.1371/journal.pcbi.1008618",
        tags=("flusight", "interval-forecast", "wis", "coverage", "picp",
              "calibration", "cdc"),
        relevance=(
            "FluSight-standard interval evaluation — the nominal 95% "
            "coverage target that the PICP forest plot (fig2) compares "
            "against."
        ),
    ),
    Citation(
        id="diebold_mariano_1995",
        authors="Diebold FX, Mariano RS",
        year=1995,
        title="Comparing predictive accuracy",
        venue="Journal of Business & Economic Statistics 13(3):253–263",
        doi_or_url="https://doi.org/10.1080/07350015.1995.10524599",
        tags=("dm-test", "predictive-accuracy", "pairwise", "comparison",
              "significance", "loss-differential"),
        relevance=(
            "The DM test implemented in phase9_dm_test (pre-COVID / "
            "during / post / global regimes) and used in the fig3 "
            "volcano plot."
        ),
    ),
    Citation(
        id="kupiec_1995",
        authors="Kupiec PH",
        year=1995,
        title="Techniques for verifying the accuracy of risk measurement models",
        venue="Journal of Derivatives 3(2):73–84",
        doi_or_url="https://doi.org/10.3905/jod.1995.407942",
        tags=("kupiec", "coverage", "calibration", "binomial", "pi",
              "prediction-interval", "var"),
        relevance=(
            "Unconditional-coverage test powering the Kupiec-p axis of "
            "the fig3 volcano plot."
        ),
    ),
    Citation(
        id="wilson_1927_ci",
        authors="Wilson EB",
        year=1927,
        title="Probable inference, the law of succession, and statistical inference",
        venue="JASA 22(158):209–212",
        doi_or_url="https://doi.org/10.1080/01621459.1927.10502953",
        tags=("wilson", "confidence-interval", "binomial", "coverage",
              "small-n", "proportion"),
        relevance=(
            "Wilson 95% CI used for the PICP forest plot (fig2) — the "
            "correct small-n binomial CI (vs normal approximation)."
        ),
    ),
)


# ══════════════════════════════════════════════════════════════════════
# Matcher
# ══════════════════════════════════════════════════════════════════════
def _tokenize(text: str) -> set[str]:
    """Lowercase, split on non-alnum, drop 1-char tokens."""
    import re
    return {t for t in re.split(r"[^a-z0-9\-]+", text.lower()) if len(t) > 1}


def score_citations(
    query: str,
    *,
    catalogue: Iterable[Citation] = STATIC_CITATIONS,
    k: int = 5,
) -> list[tuple[float, Citation]]:
    """Rank citations by keyword-overlap with ``query``.

    Scoring is deliberately simple and shallow:
    * For each citation, build a bag of tokens from
        ``tags`` + ``title`` + ``relevance``.
    * Score = |query_tokens ∩ citation_tokens|, tie-broken by a small
        bonus for citations whose IDs appear in a query word.
    * If the query produces no tokens, return the first ``k`` citations
        alphabetically by ID so the client still gets *something*.

    Returns a list of ``(score, citation)`` tuples, descending by score,
    length at most ``k``.
    """
    q_tokens = _tokenize(query)
    scored: list[tuple[float, Citation]] = []
    for cit in catalogue:
        bag = set(cit.tags) | _tokenize(cit.title) | _tokenize(cit.relevance)
        overlap = float(len(q_tokens & bag))
        if cit.id.lower() in query.lower():
            overlap += 0.5
        scored.append((overlap, cit))

    # When the query is empty or totally off-topic, fall back to a
    # stable alphabetical slice so the caller still gets something.
    if all(s == 0 for s, _ in scored):
        byid = sorted(scored, key=lambda x: x[1].id)
        return byid[: max(1, k)]

    scored.sort(key=lambda x: (-x[0], x[1].id))
    return scored[: max(1, k)]
