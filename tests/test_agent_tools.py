import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents import OpenAIChatCompletionsModel

from ocean_agent.agent import build_agent, format_run_debug
from ocean_agent.agent_tools import (
    compare_ocean_products,
    compare_ocean_products_data,
    get_ocean_product_details,
    get_ocean_product_details_data,
    search_ocean_documents,
    search_ocean_documents_data,
    search_ocean_manuals,
    search_ocean_manuals_data,
    search_ocean_products,
    search_ocean_products_data,
)
from ocean_agent.config import Settings
from ocean_agent.document_data import DOCUMENT_CHUNKS
from ocean_agent.hybrid_search import HybridDocumentSearchMatch
from ocean_agent.manual_hybrid_search import (
    ManualEvidenceGroup,
    ManualHybridSearchMatch,
)
from ocean_agent.models import (
    SourceReference,
    SourceType,
    TechnicalDocumentChunk,
    VerificationStatus,
)


def _hybrid_match(
    chunk_id: str,
    *,
    keyword_rank: int | None = 1,
    embedding_rank: int | None = 1,
) -> HybridDocumentSearchMatch:
    chunk = next(item for item in DOCUMENT_CHUNKS if item.chunk_id == chunk_id)
    methods = tuple(
        method
        for method, rank in (
            ("keyword", keyword_rank),
            ("embedding", embedding_rank),
        )
        if rank is not None
    )
    return HybridDocumentSearchMatch(
        chunk=chunk,
        fused_score=0.032787,
        keyword_score=11 if keyword_rank is not None else None,
        keyword_rank=keyword_rank,
        embedding_similarity=0.64 if embedding_rank is not None else None,
        embedding_rank=embedding_rank,
        retrieval_methods=methods,
    )


MANUAL_SOURCE = SourceReference(
    title="Official instrument guide",
    url="https://manufacturer.example/instrument-guide.pdf",
    source_type=SourceType.MANUFACTURER_OFFICIAL,
    accessed_on="2026-09-02",
    verification_status=VerificationStatus.VERIFIED,
)


def _manual_chunk(
    number: int,
    content: str,
    *,
    review_status: str = "auto_cleaned",
) -> TechnicalDocumentChunk:
    return TechnicalDocumentChunk(
        chunk_id=f"manual-tool-{number}",
        product_id="rbr-concerto3-ctd",
        document_id="rbr-instrument-guide",
        title="RBR CT/CTD Instrument Guide",
        section="Communications and power",
        content=content,
        page_number=number,
        source=MANUAL_SOURCE,
        review_status=review_status,
    )


def _manual_evidence_group() -> ManualEvidenceGroup:
    previous = _manual_chunk(10, "Connect USB-C to configure the instrument.")
    anchor = _manual_chunk(
        11,
        "USB-C power cannot be used to power the instrument while sampling.",
        review_status="needs_review",
    )
    next_chunk = _manual_chunk(12, "Use internal batteries or external power for sampling.")
    return ManualEvidenceGroup(
        anchor=ManualHybridSearchMatch(
            chunk=anchor,
            fused_score=0.0327,
            keyword_score=8,
            keyword_rank=1,
            embedding_similarity=0.71,
            embedding_rank=1,
            retrieval_methods=("keyword", "embedding"),
        ),
        previous_chunk=previous,
        next_chunk=next_chunk,
    )


class AgentToolTests(unittest.TestCase):
    def test_tool_returns_grounded_candidates_as_json(self) -> None:
        payload = json.loads(
            search_ocean_products_data(
                minimum_depth_m=5000,
                required_parameters=["temperature", "salinity", "pressure"],
            )
        )

        self.assertEqual(payload["count"], 3)
        self.assertEqual(
            {item["model"] for item in payload["products"]},
            {
                "SBE 19plus V2 SeaCAT",
                "SBE 16plus V2 SeaCAT",
                "RBRconcerto³ C.T.D",
            },
        )
        sbe16 = next(
            item for item in payload["products"] if item["model"] == "SBE 16plus V2 SeaCAT"
        )
        self.assertIn("pressure", sbe16["optional_parameters"])

    def test_tool_does_not_invent_when_no_product_matches(self) -> None:
        payload = json.loads(search_ocean_products_data(minimum_depth_m=11000))

        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["products"], [])
        self.assertIn("不要编造", payload["message"])

    def test_search_accepts_local_model_string_null_as_unspecified(self) -> None:
        payload = json.loads(
            search_ocean_products_data(
                minimum_depth_m=5000,
                required_parameters=["temperature", "salinity", "pressure"],
                deployment_type="null",
            )
        )

        self.assertEqual(payload["count"], 3)

    def test_sdk_tool_has_generated_schema(self) -> None:
        self.assertEqual(search_ocean_products.name, "search_ocean_products")
        self.assertIn("minimum_depth_m", search_ocean_products.params_json_schema["properties"])

    def test_product_details_returns_full_verified_record(self) -> None:
        payload = json.loads(get_ocean_product_details_data("SBE 19plus V2 SeaCAT"))

        self.assertTrue(payload["found"])
        self.assertEqual(payload["product"]["manufacturer"], "Sea-Bird Scientific")
        self.assertIn("measurement_specs", payload["product"])
        self.assertTrue(payload["product"]["sources"])

    def test_product_details_does_not_invent_unknown_model(self) -> None:
        payload = json.loads(get_ocean_product_details_data("不存在的 CTD-999"))

        self.assertFalse(payload["found"])
        self.assertIsNone(payload["product"])
        self.assertIn("不要", payload["message"])

    def test_details_tool_has_generated_schema(self) -> None:
        self.assertEqual(get_ocean_product_details.name, "get_ocean_product_details")
        self.assertIn("model_or_id", get_ocean_product_details.params_json_schema["properties"])

    def test_compare_products_returns_aligned_verified_records(self) -> None:
        payload = json.loads(
            compare_ocean_products_data(
                ["seabird-sbe-19plus-v2", "rbr-concerto3-ctd"]
            )
        )

        self.assertTrue(payload["can_compare"])
        self.assertEqual(len(payload["products"]), 2)
        self.assertEqual(payload["missing_product_ids"], [])
        self.assertIn("temperature", payload["common_supported_parameters"])
        self.assertTrue(all(item["sources"] for item in payload["products"]))

    def test_compare_products_reports_missing_ids_without_inventing(self) -> None:
        payload = json.loads(
            compare_ocean_products_data(
                ["seabird-sbe-19plus-v2", "not-a-real-product"]
            )
        )

        self.assertFalse(payload["can_compare"])
        self.assertEqual(payload["missing_product_ids"], ["not-a-real-product"])
        self.assertIn("不要编造", payload["message"])

    def test_compare_tool_has_generated_schema(self) -> None:
        self.assertEqual(compare_ocean_products.name, "compare_ocean_products")
        self.assertIn("product_ids", compare_ocean_products.params_json_schema["properties"])

    @patch("ocean_agent.agent_tools.hybrid_search_documents")
    def test_document_tool_returns_citable_hybrid_chunks(
        self, mock_hybrid_search
    ) -> None:
        mock_hybrid_search.return_value = [
            _hybrid_match("sbe19-interface-sampling")
        ]
        payload = json.loads(
            search_ocean_documents_data(
                "通信接口和采样率",
                model_or_id="SBE 19plus V2 SeaCAT",
            )
        )

        self.assertGreaterEqual(payload["count"], 1)
        self.assertEqual(payload["results"][0]["chunk_id"], "sbe19-interface-sampling")
        self.assertTrue(payload["results"][0]["source"]["url"])
        self.assertEqual(payload["results"][0]["keyword_rank"], 1)
        self.assertEqual(payload["results"][0]["embedding_rank"], 1)
        self.assertEqual(
            payload["results"][0]["retrieval_methods"],
            ["keyword", "embedding"],
        )
        self.assertEqual(payload["retrieval"]["mode"], "hybrid_rrf")
        self.assertEqual(
            payload["retrieval"]["answer_evidence_status"],
            "requires_content_check",
        )
        self.assertIn("不代表", payload["message"])

    @patch("ocean_agent.agent_tools.hybrid_search_documents")
    def test_document_tool_reports_missing_knowledge(
        self, mock_hybrid_search
    ) -> None:
        mock_hybrid_search.return_value = []
        payload = json.loads(
            search_ocean_documents_data(
                "故障码 E999 的维修步骤",
                model_or_id="SBE 19plus V2 SeaCAT",
            )
        )

        self.assertEqual(payload["count"], 0)
        self.assertIn("资料", payload["message"])

    @patch("ocean_agent.agent_tools.hybrid_search_documents")
    def test_document_tool_exposes_keyword_fallback(
        self, mock_hybrid_search
    ) -> None:
        mock_hybrid_search.return_value = [
            _hybrid_match(
                "rbr-ctd-interface-power-sampling",
                embedding_rank=None,
            )
        ]

        payload = json.loads(search_ocean_documents_data("RBR怎么供电？"))

        self.assertEqual(payload["results"][0]["retrieval_methods"], ["keyword"])
        self.assertIsNone(payload["results"][0]["embedding_similarity"])

    def test_document_tool_has_generated_schema(self) -> None:
        self.assertEqual(search_ocean_documents.name, "search_ocean_documents")
        self.assertIn("query", search_ocean_documents.params_json_schema["properties"])

    @patch("ocean_agent.agent_tools.hybrid_search_manual_chunks")
    def test_manual_tool_returns_grouped_citable_evidence(
        self, mock_manual_search
    ) -> None:
        mock_manual_search.return_value = [_manual_evidence_group()]

        payload = json.loads(
            search_ocean_manuals_data(
                "USB-C能否为仪器采样供电？",
                model_or_id="RBRconcerto³ C.T.D",
                limit=3,
            )
        )

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["anchor_count"], 1)
        self.assertEqual(payload["product_filter"]["product_id"], "rbr-concerto3-ctd")
        anchor = payload["evidence_groups"][0]["anchor"]
        self.assertEqual(anchor["evidence_role"], "anchor")
        self.assertEqual(anchor["page_number"], 11)
        self.assertTrue(anchor["source"]["url"])
        self.assertEqual(anchor["retrieval_methods"], ["keyword", "embedding"])
        context = payload["evidence_groups"][0]["context"]
        self.assertEqual(
            [item["evidence_role"] for item in context],
            ["previous_context", "next_context"],
        )
        self.assertIn("原始PDF", payload["evidence_groups"][0]["evidence_warning"])
        self.assertEqual(
            payload["retrieval"]["answer_evidence_status"],
            "requires_content_and_review_check",
        )
        mock_manual_search.assert_called_once_with(
            "USB-C能否为仪器采样供电？",
            product_id="rbr-concerto3-ctd",
            limit=3,
            candidate_limit=20,
        )

    @patch("ocean_agent.agent_tools.hybrid_search_manual_chunks")
    def test_manual_tool_does_not_guess_unknown_product(
        self, mock_manual_search
    ) -> None:
        payload = json.loads(
            search_ocean_manuals_data("怎么接线？", model_or_id="不存在的CTD-999")
        )

        self.assertEqual(payload["status"], "unknown_product")
        self.assertEqual(payload["anchor_count"], 0)
        self.assertIn("没有猜测", payload["message"])
        mock_manual_search.assert_not_called()

    @patch("ocean_agent.agent_tools.hybrid_search_manual_chunks")
    def test_manual_tool_reports_unavailable_local_index(
        self, mock_manual_search
    ) -> None:
        mock_manual_search.side_effect = FileNotFoundError("missing chunks")

        payload = json.loads(search_ocean_manuals_data("如何上传数据？"))

        self.assertEqual(payload["status"], "manual_index_unavailable")
        self.assertEqual(payload["evidence_groups"], [])
        self.assertIn("不得凭记忆", payload["message"])

    def test_manual_tool_has_generated_schema(self) -> None:
        self.assertEqual(search_ocean_manuals.name, "search_ocean_manuals")
        self.assertIn("query", search_ocean_manuals.params_json_schema["properties"])
        self.assertIn("limit", search_ocean_manuals.params_json_schema["properties"])

    def test_agent_registers_product_tool(self) -> None:
        agent = build_agent(Settings(_env_file=None))

        self.assertEqual(
            [item.name for item in agent.tools],
            [
                "search_ocean_products",
                "get_ocean_product_details",
                "compare_ocean_products",
                "search_ocean_manuals",
                "search_ocean_documents",
            ],
        )
        self.assertIsInstance(agent.model, OpenAIChatCompletionsModel)

    def test_debug_summary_shows_runner_loop_without_full_tool_output(self) -> None:
        result = SimpleNamespace(
            raw_responses=[
                SimpleNamespace(
                    output=[
                        SimpleNamespace(
                            type="function_call",
                            name="get_ocean_product_details",
                            arguments='{"model_or_id":"SBE 19plus V2 SeaCAT"}',
                            call_id="call-1",
                        )
                    ]
                ),
                SimpleNamespace(output=[SimpleNamespace(type="message")]),
            ],
            new_items=[
                SimpleNamespace(
                    type="tool_call_output_item",
                    raw_item={"call_id": "call-1"},
                    output='{"found":true,"product":{"model":"SBE 19plus V2 SeaCAT"}}',
                ),
            ],
            final_output="产品回答",
        )

        summary = format_run_debug(result)

        self.assertIn("[模型第 1 轮]", summary)
        self.assertIn("请求 Tool: get_ocean_product_details", summary)
        self.assertIn("SBE 19plus V2 SeaCAT", summary)
        self.assertIn("[Runner] 执行 get_ocean_product_details", summary)
        self.assertIn("[模型第 2 轮]", summary)
        self.assertIn("生成最终答案，没有新的 Tool Call", summary)
        self.assertIn("[Runner] Loop 正常结束", summary)
        self.assertIn("模型调用总轮数: 2", summary)
        self.assertIn("Tool 调用总数: 1", summary)
        self.assertNotIn('"found":true', summary)


if __name__ == "__main__":
    unittest.main()
