import json
import unittest
from types import SimpleNamespace

from agents import OpenAIChatCompletionsModel

from ocean_agent.agent import build_agent, format_run_debug
from ocean_agent.agent_tools import (
    compare_ocean_products,
    compare_ocean_products_data,
    get_ocean_product_details,
    get_ocean_product_details_data,
    search_ocean_products,
    search_ocean_products_data,
)
from ocean_agent.config import Settings


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

    def test_agent_registers_product_tool(self) -> None:
        agent = build_agent(Settings(_env_file=None))

        self.assertEqual(
            [item.name for item in agent.tools],
            [
                "search_ocean_products",
                "get_ocean_product_details",
                "compare_ocean_products",
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
