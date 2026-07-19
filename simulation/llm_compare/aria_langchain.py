"""
simulation.llm_compare.aria_langchain
======================================
ARIA-RAG — a LangChain *document-retrieval* ARIA, contrasted with the custom
numeric-grounding ARIA (``aria_grounding``).

Why this exists
---------------
The shipped ARIA (``aria_grounding``) grounds an LLM answer in the project's own
*numbers*: it extracts the gold ``key=value`` facts from a single hand-picked
result artifact, hands them to the LLM verbatim, and scores whether the answer
echoes those numbers (``numeric_grounding``). That is precise but **closed-book
on a fixed context** — there is no *retrieval*: the relevant facts are chosen by
the caller, not found by the system.

This module builds the missing half: a genuine **RAG pipeline** — a small
project-grounded corpus (real result artifacts + methodology/epidemiology notes)
→ sentence-transformers embeddings → a Chroma vector store → top-k semantic
retrieval → an Ollama LLM that answers *only from the retrieved chunks*, and
**cites its sources**. The same ``numeric_grounding`` scorer is reused so the two
ARIAs are measured on one ruler.

The honest finding (stated up front, Karpathy K-1): the corpus is *small and
project-scoped* (a handful of documents). On a tiny corpus, retrieval's marginal
value over "just give the LLM all the facts" is limited — RAG's real payoff
(scaling to a corpus too large to fit a prompt, and surfacing *which* document a
claim came from) only partly materialises here. What RAG adds even at this scale
is **(a) source attribution / citations** and **(b) query-driven selection** of
context instead of a caller-fixed context. We report that difference rather than
overclaiming a quality win.

Deep module (D-4): one public entry, :func:`build_rag` (corpus → retriever →
chain), plus :func:`run_demo` (the producer) and :func:`compare_to_custom_aria`
(the head-to-head). Heavy LangChain/Chroma/embedding wiring is encapsulated;
callers see a 3-function surface.

Install (pip, NOT ``uv sync`` — would break requirements.lock):
    .venv/bin/python -m pip install langchain langchain-community \
        langchain-ollama langchain-chroma chromadb ollama

CLI:
    .venv/bin/python -m simulation.llm_compare.aria_langchain --model qwen2.5:3b
    .venv/bin/python -m simulation.llm_compare.aria_langchain --dry-run   # no LLM
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

# LangChain emits cosmetic deprecation noise for the community HuggingFace
# embeddings shim; silence only that so the demo stdout stays readable.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain.*")
warnings.filterwarnings("ignore", message=".*HuggingFaceEmbeddings.*")

from .aria_grounding import load_real_context, numeric_grounding

__all__ = [
    "DEFAULT_MODEL", "DEFAULT_EMBED_MODEL", "RAG_PROMPT",
    "build_corpus", "build_rag", "AriaRagChain",
    "run_demo", "compare_to_custom_aria",
]

DEFAULT_MODEL = "qwen2.5:3b"            # fast local Ollama LLM (mistral:7b = --model)
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # offline, 384-d, cached

# The RAG instruction: answer ONLY from retrieved context, cite sources, refuse
# to invent numbers. Mirrors GROUNDING_PROMPT's grounding discipline so the two
# ARIAs are held to the same "no hallucinated numbers" bar.
RAG_PROMPT = (
    "당신은 역학자에게 시뮬레이션 결과를 해석해 주는 자문가입니다.\n"
    "아래 [검색된 문맥]에 있는 내용만 근거로 한국어 2~4문장으로 답하세요.\n"
    "문맥에 없는 수치는 절대 지어내지 마세요. 답변 끝에 사용한 출처를 "
    "[출처: <source>] 형식으로 인용하세요.\n\n"
    "[검색된 문맥]\n{context}\n\n[질문] {question}\n\n[답변]"
)


# ── corpus ────────────────────────────────────────────────────────────────────
def build_corpus(*, root: str | None = None) -> list[dict]:
    """Assemble the project-grounded RAG corpus as plain dicts.

    The corpus has two strata, both project-scoped (NOT a large external KB):

      * **result artifacts** — the SAME real numbers the custom ARIA grounds on,
        pulled via :func:`aria_grounding.load_real_context` (ABM forward
        validation + ABM real-wave fit). This makes the two ARIAs comparable: a
        question about ``forward_r2`` can be answered by *retrieving* the
        artifact chunk, exactly as the custom ARIA answers from the fixed fact
        list.
      * **methodology / epidemiology notes** — short hand-authored explanations
        of WIS, conformal prediction intervals, the SEIR-V-D compartments, and
        ILI surveillance caveats. These give the retriever something to *choose
        between*, so a question about "what is WIS" pulls the method note, not
        the ABM numbers.

    Args:
        root: results root forwarded to ``load_real_context`` (default
            ``simulation/results``). If an artifact is missing, that document is
            skipped (the methodology stratum still yields a usable corpus).

    Returns:
        ``[{id, text, source, kind, facts}]`` — ``facts`` is the gold
        ``key=value`` list for artifact docs (``[]`` for method notes), used by
        the grounding scorer. Never empty: method notes are always present.

    Side effects: reads result JSON via ``load_real_context`` (no DB, no writes).
    """
    docs: list[dict] = []

    # stratum 1 — real result artifacts (shared ground truth with custom ARIA)
    for which in ("identifiability", "abm"):
        try:
            c = load_real_context(which, root=root)
        except FileNotFoundError:
            continue  # honest: skip a missing artifact rather than fabricate one
        docs.append({
            "id": f"artifact::{c['id']}",
            "text": c["context"],
            "source": c["source"],
            "kind": "result_artifact",
            "facts": c["facts"],
        })

    # stratum 2 — methodology / epidemiology knowledge (project-authored notes)
    notes = [
        ("method::WIS",
         "WIS(가중 구간 점수, Weighted Interval Score)는 확률예측 평가지표로, "
         "여러 예측구간의 coverage와 sharpness를 결합한다. WIS가 낮을수록 좋다. "
         "본 프로젝트의 챔피언 선정 1차 기준이 OOF-WIS이며, 점예측 정확도(R²)와 "
         "구간 보정(PICP)을 동시에 반영한다.",
         "methodology/metrics.md"),
        ("method::conformal",
         "Conformal prediction(등각 예측)은 분포가정 없이 예측구간의 유한표본 "
         "coverage를 보장하는 기법이다. Adaptive conformal은 시간에 따라 잔차 "
         "분위수를 갱신해 분포이동에도 명목 coverage(예: 90%)를 유지한다. 본 "
         "프로젝트는 adaptive conformal로 전 모델 PI를 0.67→0.90으로 보정했다.",
         "methodology/conformal.md"),
        ("epi::SEIR-V-D",
         "SEIR-V-D는 감수성(S)·잠복(E)·감염(I)·회복(R)에 백신접종(V)과 사망(D) "
         "구획을 더한 구획 모형이다. 메타개체군(metapopulation) 버전은 25개 자치구를 "
         "통근 이동으로 결합해 지역 간 전파를 모형화한다. 행동(behavior) 모듈은 "
         "유병률에 반응해 접촉을 조절하는 prevalence-dependent 항을 더한다.",
         "epidemiology/seir.md"),
        ("epi::ILI",
         "ILI(인플루엔자 유사 증상, Influenza-Like Illness) rate는 표본감시 "
         "의료기관에서 보고한 발열+호흡기 증상 환자 비율이다. 실제 발생률이 아니라 "
         "감시 신호이며 보고지연·과소확인 편향이 있다. 본 프로젝트의 1차 예측 "
         "대상은 서울 주간 ILI rate이다.",
         "epidemiology/ili.md"),
        ("method::ABM-behavior",
         "행동 기반 ABM(agent-based model)에서 behavior-ON은 행위자가 유병률에 "
         "반응해 접촉을 줄이는 설정이고 behavior-OFF(static)는 반응이 없는 설정이다. "
         "전향 검증에서 behavior-ON 전향 R²가 OFF보다 높으면 행동 항이 예측에 "
         "기여함을 뜻한다. ABM은 forecaster가 아니라 메커니즘·반사실(counterfactual) "
         "엔진으로 쓰인다.",
         "methodology/abm.md"),
    ]
    for nid, text, source in notes:
        docs.append({"id": nid, "text": text, "source": source,
                     "kind": "knowledge_note", "facts": []})
    return docs


# ── RAG chain ─────────────────────────────────────────────────────────────────
class AriaRagChain:
    """A built RAG pipeline: retriever + (optional) Ollama LLM over a Chroma store.

    Small interface (``query``, ``retrieve``), rich implementation (embeds the
    corpus, indexes it in an in-memory Chroma collection, and runs an LCEL
    retrieve→prompt→LLM→parse chain). Construct via :func:`build_rag`, not
    directly.

    Attributes:
        retriever: the Chroma similarity retriever (top-k).
        corpus: the ``build_corpus`` docs (for fact lookup / reporting).
        model: Ollama model id, or ``None`` in dry-run (retrieval only).
    """

    def __init__(self, retriever, corpus: list[dict], model: str | None,
                 *, chain=None, by_source: dict | None = None):
        self.retriever = retriever
        self.corpus = corpus
        self.model = model
        self._chain = chain
        self._by_source = by_source or {d["source"]: d for d in corpus}

    def retrieve(self, question: str) -> list[dict]:
        """Top-k semantic retrieval for ``question``.

        Returns:
            ``[{text, source, kind}]`` for the retrieved chunks, best-first.

        Side effects: embeds the query (CPU) + Chroma similarity search. No DB.
        """
        hits = self.retriever.invoke(question)
        out = []
        for h in hits:
            src = h.metadata.get("source", "?")
            ref = self._by_source.get(src, {})
            out.append({"text": h.page_content, "source": src,
                        "kind": ref.get("kind", "?")})
        return out

    def query(self, question: str) -> dict:
        """Full RAG: retrieve → (LLM answer | dry-run stitched context).

        Args:
            question: a natural-language question for ARIA.

        Returns:
            ``{question, answer, retrieved:[{source,kind,text}], facts}`` where
            ``facts`` aggregates the gold numeric facts of every retrieved
            *artifact* chunk (used by the grounding scorer). In dry-run
            (``model is None``) ``answer`` is the stitched retrieved context — no
            LLM call — so retrieval is testable offline.

        Side effects: query embed + Chroma search; one Ollama HTTP call iff a
            model is set. Never raises on LLM failure (records ``[LLM error: …]``).
        """
        retrieved = self.retrieve(question)
        # aggregate gold facts from retrieved artifact chunks (for scoring)
        facts: list[str] = []
        for r in retrieved:
            facts.extend(self._by_source.get(r["source"], {}).get("facts", []))

        if self._chain is None:  # dry-run: retrieval only, no LLM
            ctx = "\n".join(f"[{r['source']}] {r['text']}" for r in retrieved)
            return {"question": question, "answer": ctx, "retrieved": retrieved,
                    "facts": facts, "dry_run": True}
        try:
            answer = self._chain.invoke(question)
        except Exception as e:  # noqa: BLE001 — LLM/transport failure is data, not a crash
            answer = f"[LLM error: {type(e).__name__}: {str(e)[:160]}]"
        return {"question": question, "answer": (answer or "").strip(),
                "retrieved": retrieved, "facts": facts, "dry_run": False}


def build_rag(*, model: str | None = DEFAULT_MODEL,
              embed_model: str = DEFAULT_EMBED_MODEL, top_k: int = 3,
              root: str | None = None, temperature: float = 0.2) -> AriaRagChain:
    """Build the ARIA-RAG pipeline: corpus → embeddings → Chroma → retriever → LLM.

    Args:
        model: Ollama model id for the answer LLM (e.g. ``qwen2.5:3b`` /
            ``mistral:7b``). Pass ``None`` for a **dry-run** retriever-only chain
            (no Ollama needed) — used by tests and ``--dry-run``.
        embed_model: sentence-transformers model for the vector store (offline,
            already cached; default MiniLM-L6 = 384-d).
        top_k: number of chunks retrieved per query.
        root: results root for the artifact stratum of the corpus.
        temperature: LLM sampling temperature (low = grounded).

    Returns:
        A ready :class:`AriaRagChain`.

    Raises:
        ImportError: if the LangChain/Chroma stack is not installed (the message
            states the exact ``pip install`` line — NOT ``uv sync``).

    Performance: embedding the ~7-doc corpus is a one-off ~1-2 s after the
        MiniLM weights are cached; each query is a sub-second embed + search plus
        one Ollama call (~2-30 s depending on model).
    Side effects: loads the embedding model into memory; builds an in-memory
        Chroma collection (no on-disk DB, no sqlite file written).
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import PromptTemplate
        from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    except ImportError as e:  # pragma: no cover - environment guard
        raise ImportError(
            "ARIA-RAG needs the LangChain stack. Install with pip (NOT `uv sync` "
            "— that rewrites requirements.lock and removes mlx-lm/transformers, "
            "breaking ARIA):\n"
            "  .venv/bin/python -m pip install langchain langchain-community "
            "langchain-ollama langchain-chroma chromadb ollama"
        ) from e

    corpus = build_corpus(root=root)
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    lc_docs = [Document(page_content=d["text"],
                        metadata={"source": d["source"], "kind": d["kind"], "id": d["id"]})
               for d in corpus]
    # in-memory Chroma (no persist_directory) → no sqlite/.db artifact on disk
    store = Chroma.from_documents(lc_docs, embedding=embeddings,
                                  collection_name="aria_rag_demo")
    retriever = store.as_retriever(search_kwargs={"k": top_k})

    if model is None:
        return AriaRagChain(retriever, corpus, None)

    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(model=model, temperature=temperature)
    prompt = PromptTemplate.from_template(RAG_PROMPT)

    def _format(docs) -> str:
        return "\n\n".join(f"[출처: {d.metadata.get('source','?')}] {d.page_content}"
                           for d in docs)

    # LCEL: {context: retriever|format, question: passthrough} → prompt → llm → str
    chain = (
        {"context": retriever | RunnableLambda(_format),
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return AriaRagChain(retriever, corpus, model, chain=chain)


# ── default demo questions ────────────────────────────────────────────────────
DEMO_QUESTIONS = [
    "ABM 전향 검증에서 behavior-ON과 OFF 중 어느 쪽 전향 R²가 더 높았나요?",
    "WIS는 무엇이고 챔피언 선정에서 어떤 역할을 하나요?",
    "ABM 실파동 적합에서 적응 R²와 정적 R²는 각각 얼마였나요?",
    "Conformal prediction은 예측구간 coverage를 어떻게 보장하나요?",
    "보정된 행동 파라미터 alpha와 theta 값은 무엇이며 무슨 의미인가요?",
]


# ── producers ─────────────────────────────────────────────────────────────────
def run_demo(*, model: str | None = DEFAULT_MODEL, top_k: int = 3,
             questions: list[str] | None = None, root: str | None = None) -> dict:
    """Run the RAG demo over the demo questions and score grounding per answer.

    Args:
        model: Ollama model (``None`` = dry-run, retrieval only).
        top_k: chunks retrieved per query.
        questions: override the default demo questions.
        root: results root for the corpus.

    Returns:
        ``{model, embed_model, top_k, n_docs, corpus_sources, per_query:[...],
        grounding_summary}`` — each ``per_query`` carries the question, retrieved
        sources, the answer, and its ``numeric_grounding`` (fact_recall /
        n_spurious vs the retrieved artifacts' gold facts).

    Side effects: builds the chain (loads embed model); one Ollama call per
        question if a model is set. No DB, no sqlite written.
    """
    rag = build_rag(model=model, top_k=top_k, root=root)
    qs = questions or DEMO_QUESTIONS
    per_query, recalls, spurious = [], [], 0
    for q in qs:
        res = rag.query(q)
        ng = numeric_grounding(res["answer"], res["facts"]) if res["facts"] else {
            "fact_recall": None, "n_gold_cited": 0, "n_gold": 0, "n_spurious": 0}
        if ng["fact_recall"] is not None:
            recalls.append(ng["fact_recall"])
            spurious += ng["n_spurious"]
        per_query.append({
            "question": q,
            "retrieved": [{"source": r["source"], "kind": r["kind"]}
                          for r in res["retrieved"]],
            "answer": res["answer"],
            "grounding": ng,
        })
    grounding_summary = {
        "mean_fact_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "n_spurious_total": spurious,
        "n_scored": len(recalls),
    }
    corpus = rag.corpus
    return {
        "approach": "langchain_rag",
        "model": model or "(dry-run, retrieval only)",
        "embed_model": DEFAULT_EMBED_MODEL,
        "top_k": top_k,
        "n_docs": len(corpus),
        "corpus_sources": [{"source": d["source"], "kind": d["kind"]} for d in corpus],
        "per_query": per_query,
        "grounding_summary": grounding_summary,
    }


def compare_to_custom_aria() -> dict:
    """Structured contrast: LangChain-RAG ARIA vs custom numeric-grounding ARIA.

    This is the honest, code-free comparison the task asks for — a fixed
    description of *what differs*, *what RAG adds*, and *the limits at this corpus
    size*. It does not re-run either system (the demo JSON carries the live
    numbers); it is the qualitative axis that accompanies them.

    Returns:
        ``{dimensions:[{dimension, custom_aria, langchain_rag}], rag_adds[],
        custom_keeps[], honest_limits[]}``.

    Side effects: none. Never raises.
    """
    return {
        "dimensions": [
            {"dimension": "context selection",
             "custom_aria": "caller fixes the context (one chosen artifact's facts)",
             "langchain_rag": "query-driven: retriever PICKS top-k chunks by semantic similarity"},
            {"dimension": "knowledge source",
             "custom_aria": "a single result artifact's key=value facts",
             "langchain_rag": "a multi-doc corpus (artifacts + methodology + epi notes)"},
            {"dimension": "source attribution",
             "custom_aria": "none — one fixed source, not cited per-claim",
             "langchain_rag": "answer cites [출처: <source>]; retrieved docs are listed"},
            {"dimension": "grounding check",
             "custom_aria": "deterministic numeric_grounding (fact_recall / n_spurious)",
             "langchain_rag": "same numeric_grounding REUSED on the retrieved-fact set"},
            {"dimension": "scaling",
             "custom_aria": "limited — all facts must fit one hand-picked context",
             "langchain_rag": "scales to a corpus too large to fit a prompt (retrieve subset)"},
            {"dimension": "determinism",
             "custom_aria": "fully deterministic (no embeddings, no retrieval)",
             "langchain_rag": "retrieval deterministic; LLM answer is sampled"},
        ],
        "rag_adds": [
            "query-driven retrieval (the system finds the relevant context, not the caller)",
            "per-answer source citation / provenance",
            "extensibility to a large corpus where prompt-stuffing all facts is infeasible",
            "ability to answer methodology questions (WIS, conformal) that have no single number",
        ],
        "custom_keeps": [
            "deterministic, leak-free numeric grounding (no sampling variance)",
            "no embedding model / vector store dependency (lighter, fewer moving parts)",
            "Self-Ask decomposition over the exact gold facts (aria_grounding)",
        ],
        "honest_limits": [
            "corpus is small and project-scoped (~7 docs) — retrieval's marginal "
            "value over giving the LLM all facts is limited at this scale",
            "RAG's real payoff (large-corpus selection) is only partly realised here; "
            "the demonstrable wins are citation + query-driven selection, not a "
            "grounding-accuracy jump",
            "the LLM answer is sampled, so its numeric grounding can vary run-to-run, "
            "unlike the custom ARIA's deterministic score",
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="ARIA-RAG (LangChain document retrieval) demo + custom-ARIA contrast")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Ollama LLM id (e.g. qwen2.5:3b, mistral:7b)")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true",
                    help="retrieval only, no Ollama LLM (offline-safe)")
    ap.add_argument("--out", default="simulation/results/aria_langchain_demo.json")
    args = ap.parse_args(argv)

    model = None if args.dry_run else args.model
    demo = run_demo(model=model, top_k=args.top_k)
    comparison = compare_to_custom_aria()
    payload = {"demo": demo, "comparison_vs_custom_aria": comparison}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"ARIA-RAG demo — model={demo['model']}, embed={demo['embed_model']}, "
          f"corpus={demo['n_docs']} docs, top_k={demo['top_k']}")
    for q in demo["per_query"]:
        srcs = ", ".join(r["source"] for r in q["retrieved"])
        gr = q["grounding"]
        print(f"\nQ: {q['question']}")
        print(f"  retrieved: {srcs}")
        print(f"  answer: {q['answer'][:220]}{'…' if len(q['answer']) > 220 else ''}")
        if gr["fact_recall"] is not None:
            print(f"  grounding: fact_recall={gr['fact_recall']} "
                  f"(cited {gr['n_gold_cited']}/{gr['n_gold']}, spurious={gr['n_spurious']})")
    gs = demo["grounding_summary"]
    print(f"\nGrounding summary: mean_fact_recall={gs['mean_fact_recall']} "
          f"(scored {gs['n_scored']} q, spurious_total={gs['n_spurious_total']})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
