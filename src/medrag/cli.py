"""Command-line interface: `medrag download | ingest | ask | eval | serve`."""

from __future__ import annotations

import argparse
import json
import logging
import textwrap
from dataclasses import asdict

from medrag.config import PROJECT_ROOT, Settings
from medrag.datasets import download_datasets, load_liveqa_benchmark
from medrag.evaluation import evaluate, evaluate_benchmark, load_qa
from medrag.pipeline import RAGPipeline


def _download(args: argparse.Namespace, settings: Settings) -> None:
    download_datasets(settings.external_dir)


def _ingest(args: argparse.Namespace, settings: Settings) -> None:
    pipeline = RAGPipeline.build(settings)
    print(f"Indexed {len(pipeline.retriever.store)} chunks -> {settings.index_dir}")


def _ask(args: argparse.Namespace, settings: Settings) -> None:
    pipeline = RAGPipeline.load(settings)
    answer = pipeline.ask(args.question, top_k=args.top_k, mode=args.mode, rerank=args.rerank)
    if answer.emergency_flags:
        print("!! URGENT: possible emergency symptoms detected:", ", ".join(answer.emergency_flags))
        print()
    print("Answer:\n" + textwrap.fill(answer.answer, width=100))
    print("\nSources:")
    for s in answer.sources:
        section = f" - {s.section}" if s.section else ""
        print(f"  [{s.index}] {s.source}{section} (score={s.score})")
        if s.url:
            print(f"      {s.url}")
    print(f"\n{answer.disclaimer}")


def _eval_liveqa(pipeline: RAGPipeline, settings: Settings, args: argparse.Namespace) -> None:
    queries = load_liveqa_benchmark(settings.external_dir / "liveqa")
    configs = [("sparse", False), ("dense", False), ("hybrid", False)]
    if pipeline.retriever.reranker is not None:
        configs.append(("hybrid", True))
    results = [
        evaluate_benchmark(
            pipeline.retriever, queries, mode=mode, rerank=rerank,
            min_grade=args.min_grade, use_paraphrase=args.paraphrase,
        )
        for mode, rerank in configs
    ]
    kind = "NIST paraphrase" if args.paraphrase else "original consumer"
    print(f"\nTREC 2017 LiveQA-Med ({kind} questions, n={results[0].n}, relevant = grade >= {args.min_grade})")
    print("| Retriever | Hit@1 | Hit@5 | MRR@10 | nDCG@10 | ms/query |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r.name} | {r.hit_at_1:.3f} | {r.hit_at_5:.3f} | {r.mrr_at_10:.3f} | {r.ndcg_at_10:.3f} | {r.ms_per_query:.0f} |")
    if args.output:
        out = PROJECT_ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "benchmark": "TREC-2017-LiveQA-Med",
            "questions": kind,
            "min_grade": args.min_grade,
            "embedding_model": pipeline.retriever.embedder.name,
            "reranker": getattr(pipeline.retriever.reranker, "name", None),
            "chunks_indexed": len(pipeline.retriever.store),
            "results": [asdict(r) for r in results],
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved -> {out}")


def _eval(args: argparse.Namespace, settings: Settings) -> None:
    pipeline = RAGPipeline.load(settings)
    if args.benchmark == "liveqa":
        _eval_liveqa(pipeline, settings, args)
        return
    qa = load_qa(args.qa_file)
    print(f"{'mode':<8} {'k':>3} {'HitRate':>8} {'MRR':>7}   (n={len(qa)})")
    for mode in ("sparse", "dense", "hybrid"):
        r = evaluate(pipeline.retriever, qa, k=args.top_k, mode=mode)
        print(f"{r.mode:<8} {r.k:>3} {r.hit_rate:>8.3f} {r.mrr:>7.3f}")


def _serve(args: argparse.Namespace, settings: Settings) -> None:
    import uvicorn

    uvicorn.run("medrag.api:app", host=args.host, port=args.port)


def main() -> None:
    parser = argparse.ArgumentParser(prog="medrag", description="Healthcare RAG assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="Download MedQuAD + LiveQA datasets").set_defaults(func=_download)
    sub.add_parser("ingest", help="Build the vector index").set_defaults(func=_ingest)

    ask = sub.add_parser("ask", help="Ask a question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None)
    ask.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default=None)
    ask.add_argument("--no-rerank", dest="rerank", action="store_false", default=None)
    ask.set_defaults(func=_ask)

    ev = sub.add_parser("eval", help="Evaluate retrieval quality")
    ev.add_argument("--benchmark", choices=["liveqa", "local"], default="liveqa")
    ev.add_argument("--min-grade", type=int, default=2, help="LiveQA: 1=Related, 2=Incomplete, 3=Excellent")
    ev.add_argument("--paraphrase", action="store_true", help="LiveQA: use NIST paraphrased questions")
    ev.add_argument("--output", default=None, help="LiveQA: save results JSON (relative to project root)")
    ev.add_argument("--qa-file", default=PROJECT_ROOT / "data" / "eval" / "qa.jsonl")
    ev.add_argument("--top-k", type=int, default=5)
    ev.set_defaults(func=_eval)

    serve = sub.add_parser("serve", help="Run the REST API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_serve)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args.func(args, Settings.from_env())


if __name__ == "__main__":
    main()
