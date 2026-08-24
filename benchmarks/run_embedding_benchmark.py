"""轮流评测开源 Embedding 模型的中英跨语言海洋设备检索能力。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "benchmarks" / "ocean_embedding_cases.json"
DEFAULT_CACHE = PROJECT_ROOT.parent / ".cache" / "huggingface"
DEFAULT_RESULTS = PROJECT_ROOT / "benchmarks" / "results"

QUERY_INSTRUCTION = (
    "Given a marine equipment technical query, retrieve passages that answer the query."
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    query_template: str
    document_template: str = "{text}"
    trust_remote_code: bool = False


MODEL_SPECS = {
    "qwen3-0.6b": ModelSpec(
        key="qwen3-0.6b",
        model_id="Qwen/Qwen3-Embedding-0.6B",
        query_template=f"Instruct: {QUERY_INSTRUCTION}\nQuery:{{text}}",
    ),
    "bge-m3": ModelSpec(
        key="bge-m3",
        model_id="BAAI/bge-m3",
        query_template="{text}",
    ),
    "multilingual-e5": ModelSpec(
        key="multilingual-e5",
        model_id="intfloat/multilingual-e5-large-instruct",
        query_template=f"Instruct: {QUERY_INSTRUCTION}\nQuery: {{text}}",
    ),
    "nomic-v2": ModelSpec(
        key="nomic-v2",
        model_id="nomic-ai/nomic-embed-text-v2-moe",
        query_template="search_query: {text}",
        document_template="search_document: {text}",
        trust_remote_code=True,
    ),
}


def load_dataset(
    path: Path,
    _loading: frozenset[Path] = frozenset(),
) -> dict[str, Any]:
    path = path.resolve()
    if path in _loading:
        raise ValueError(f"评测集 extends 出现循环引用: {path}")
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    if data.get("extends"):
        base_path = (path.parent / data["extends"]).resolve()
        base = load_dataset(base_path, _loading | {path})
        data = {
            **data,
            "documents": [*base["documents"], *data.get("documents", [])],
            "cases": [*base["cases"], *data.get("cases", [])],
        }

    if not data.get("documents") or not data.get("cases"):
        raise ValueError("评测集必须包含 documents 和 cases")
    document_ids = [item["id"] for item in data["documents"]]
    case_ids = [item["id"] for item in data["cases"]]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("document id 不能重复")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case id 不能重复")

    known_documents = set(document_ids)
    for case in data["cases"]:
        candidates = case.get("candidate_ids", [])
        relevant = set(case.get("relevant_ids", []))
        if len(candidates) < 2:
            raise ValueError(f"{case['id']} 至少需要两个候选资料")
        if len(candidates) != len(set(candidates)):
            raise ValueError(f"{case['id']} 的候选资料不能重复")
        if not set(candidates) <= known_documents:
            raise ValueError(f"{case['id']} 引用了不存在的候选资料")
        if not relevant <= set(candidates):
            raise ValueError(f"{case['id']} 的正确资料必须属于候选资料")
    return data


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int,
) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def evaluate_model(
    spec: ModelSpec,
    dataset: dict[str, Any],
    *,
    cache_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    load_started = time.perf_counter()
    model = SentenceTransformer(
        spec.model_id,
        device=device,
        cache_folder=str(cache_dir),
        trust_remote_code=spec.trust_remote_code,
        model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {},
    )
    load_seconds = time.perf_counter() - load_started

    documents = dataset["documents"]
    document_by_id = {item["id"]: item for item in documents}
    document_texts = [
        spec.document_template.format(text=item["text"]) for item in documents
    ]
    query_texts = [
        spec.query_template.format(text=item["query"]) for item in dataset["cases"]
    ]

    encode_started = time.perf_counter()
    document_vectors = encode_texts(model, document_texts, batch_size=batch_size)
    query_vectors = encode_texts(model, query_texts, batch_size=batch_size)
    encode_seconds = time.perf_counter() - encode_started
    vector_by_id = {
        item["id"]: vector
        for item, vector in zip(documents, document_vectors, strict=True)
    }

    case_results = []
    positive_cases = []
    no_answer_max_scores = []
    by_direction: dict[str, list[dict[str, Any]]] = {}

    for case, query_vector in zip(dataset["cases"], query_vectors, strict=True):
        ranked = sorted(
            (
                {
                    "document_id": document_id,
                    "score": float(query_vector @ vector_by_id[document_id]),
                    "text": document_by_id[document_id]["text"],
                }
                for document_id in case["candidate_ids"]
            ),
            key=lambda item: (-item["score"], item["document_id"]),
        )
        relevant_ids = set(case["relevant_ids"])
        result: dict[str, Any] = {
            "case_id": case["id"],
            "direction": case["direction"],
            "category": case.get("category", "baseline"),
            "query": case["query"],
            "relevant_ids": case["relevant_ids"],
            "ranking": ranked,
        }

        if relevant_ids:
            first_relevant_rank = next(
                index
                for index, item in enumerate(ranked, start=1)
                if item["document_id"] in relevant_ids
            )
            best_relevant = max(
                item["score"]
                for item in ranked
                if item["document_id"] in relevant_ids
            )
            best_negative = max(
                item["score"]
                for item in ranked
                if item["document_id"] not in relevant_ids
            )
            result.update(
                {
                    "first_relevant_rank": first_relevant_rank,
                    "top1": first_relevant_rank == 1,
                    "hit_at_3": first_relevant_rank <= 3,
                    "recall_at_3": sum(
                        item["document_id"] in relevant_ids for item in ranked[:3]
                    )
                    / len(relevant_ids),
                    "reciprocal_rank": 1.0 / first_relevant_rank,
                    "ndcg_at_3": sum(
                        1.0 / math.log2(rank + 1)
                        for rank, item in enumerate(ranked[:3], start=1)
                        if item["document_id"] in relevant_ids
                    )
                    / sum(
                        1.0 / math.log2(rank + 1)
                        for rank in range(1, min(len(relevant_ids), 3) + 1)
                    ),
                    "margin": best_relevant - best_negative,
                }
            )
            positive_cases.append(result)
            by_direction.setdefault(case["direction"], []).append(result)
        else:
            max_score = ranked[0]["score"]
            result["no_answer_max_score"] = max_score
            no_answer_max_scores.append(max_score)

        case_results.append(result)

    def summarize(items: list[dict[str, Any]]) -> dict[str, float | int]:
        return {
            "count": len(items),
            "top1_accuracy": sum(item["top1"] for item in items) / len(items),
            "hit_at_3": sum(item["hit_at_3"] for item in items) / len(items),
            "recall_at_3": sum(item["recall_at_3"] for item in items) / len(items),
            "mrr": sum(item["reciprocal_rank"] for item in items) / len(items),
            "ndcg_at_3": sum(item["ndcg_at_3"] for item in items) / len(items),
            "mean_margin": sum(item["margin"] for item in items) / len(items),
        }

    dimension = int(document_vectors.shape[1])
    return {
        "model": asdict(spec),
        "dataset": {
            "description": dataset.get("description", ""),
            "document_count": len(documents),
            "case_count": len(dataset["cases"]),
            "answerable_count": len(positive_cases),
            "no_answer_count": len(no_answer_max_scores),
        },
        "device": device,
        "embedding_dimension": dimension,
        "load_seconds": load_seconds,
        "encode_seconds": encode_seconds,
        "evaluated_at": datetime.now().astimezone().isoformat(),
        "summary": summarize(positive_cases),
        "by_direction": {
            direction: summarize(items) for direction, items in by_direction.items()
        },
        "by_category": {
            category: summarize(
                [item for item in positive_cases if item["category"] == category]
            )
            for category in sorted({item["category"] for item in positive_cases})
        },
        "no_answer": {
            "count": len(no_answer_max_scores),
            "mean_max_score": (
                sum(no_answer_max_scores) / len(no_answer_max_scores)
                if no_answer_max_scores
                else None
            ),
            "scores": no_answer_max_scores,
            "note": "无答案分数只用于后续选择阈值，不直接计入Top-1。",
        },
        "cases": case_results,
    }


def print_summary(result: dict[str, Any]) -> None:
    summary = result["summary"]
    print(f"模型: {result['model']['model_id']}")
    print(f"设备: {result['device']}")
    print(f"向量维度: {result['embedding_dimension']}")
    print(f"模型加载: {result['load_seconds']:.2f} 秒")
    print(f"编码耗时: {result['encode_seconds']:.2f} 秒")
    print(f"Top-1: {summary['top1_accuracy']:.1%}")
    print(f"Hit@3: {summary['hit_at_3']:.1%}")
    print(f"Recall@3: {summary['recall_at_3']:.1%}")
    print(f"MRR: {summary['mrr']:.4f}")
    print(f"nDCG@3: {summary['ndcg_at_3']:.4f}")
    print(f"平均领先分差: {summary['mean_margin']:.6f}")
    print("\n逐方向:")
    for direction, values in result["by_direction"].items():
        print(
            f"  {direction}: Top-1={values['top1_accuracy']:.1%}, "
            f"nDCG@3={values['ndcg_at_3']:.4f}, "
            f"margin={values['mean_margin']:.6f}"
        )
    print("\n逐难度类型:")
    for category, values in result["by_category"].items():
        print(
            f"  {category}: Top-1={values['top1_accuracy']:.1%}, "
            f"nDCG@3={values['ndcg_at_3']:.4f}, "
            f"margin={values['mean_margin']:.6f}"
        )
    print("\n失败案例:")
    failures = [item for item in result["cases"] if item.get("top1") is False]
    if not failures:
        print("  无")
    for item in failures:
        print(
            f"  {item['case_id']}: 期望={item['relevant_ids']}, "
            f"实际第一={item['ranking'][0]['document_id']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="运行海洋设备Embedding领域评测")
    parser.add_argument("--model", choices=MODEL_SPECS, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size 必须大于0")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = evaluate_model(
        MODEL_SPECS[args.model],
        load_dataset(args.dataset),
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
    )
    output_path = args.output_dir / f"{args.model}.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_summary(result)
    print(f"\n完整结果: {output_path}")


if __name__ == "__main__":
    main()
