"""评测集结构测试；在 embedding-benchmark 环境中运行。"""

import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.run_embedding_benchmark import load_dataset


BENCHMARK_DIR = Path(__file__).resolve().parent


class BenchmarkDatasetTests(unittest.TestCase):
    def test_stress_dataset_extends_base_dataset(self) -> None:
        dataset = load_dataset(BENCHMARK_DIR / "ocean_embedding_stress_cases.json")

        self.assertEqual(len(dataset["documents"]), 28)
        self.assertEqual(len(dataset["cases"]), 40)
        self.assertEqual(
            sum(bool(case["relevant_ids"]) for case in dataset["cases"]),
            32,
        )

    def test_relevant_document_must_be_a_candidate(self) -> None:
        invalid_dataset = {
            "documents": [
                {"id": "doc-a", "text": "A"},
                {"id": "doc-b", "text": "B"},
                {"id": "doc-c", "text": "C"},
            ],
            "cases": [
                {
                    "id": "case-a",
                    "query": "query",
                    "direction": "zh_to_zh",
                    "candidate_ids": ["doc-a", "doc-b"],
                    "relevant_ids": ["doc-c"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(invalid_dataset), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "正确资料必须属于候选资料"):
                load_dataset(path)

    def test_duplicate_document_id_is_rejected(self) -> None:
        invalid_dataset = {
            "documents": [
                {"id": "duplicate", "text": "A"},
                {"id": "duplicate", "text": "B"},
            ],
            "cases": [
                {
                    "id": "case-a",
                    "query": "query",
                    "direction": "zh_to_zh",
                    "candidate_ids": ["duplicate", "duplicate"],
                    "relevant_ids": [],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(invalid_dataset), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "document id 不能重复"):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
