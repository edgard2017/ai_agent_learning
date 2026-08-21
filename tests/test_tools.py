import unittest

from ocean_agent.models import DeploymentType, SourceType
from ocean_agent.product_data import PRODUCTS
from ocean_agent.tools import get_product_spec, search_products


class ProductDataTests(unittest.TestCase):
    def test_catalog_has_four_verified_public_products(self) -> None:
        self.assertEqual(len(PRODUCTS), 4)
        for product in PRODUCTS:
            self.assertEqual(product.data_scope, "third_party_public_product")
            self.assertTrue(product.sources)
            self.assertTrue(
                all(
                    source.source_type is SourceType.MANUFACTURER_OFFICIAL
                    for source in product.sources
                )
            )

    def test_search_by_depth_and_chinese_parameter_aliases(self) -> None:
        matches = search_products(
            minimum_depth_m=5000,
            required_parameters=["温度", "盐度", "压力"],
        )
        models = {product.model for product in matches}
        self.assertIn("SBE 19plus V2 SeaCAT", models)
        self.assertIn("RBRconcerto³ C.T.D", models)
        self.assertNotIn("SBE 37 MicroCAT", models)

    def test_search_by_deployment_type(self) -> None:
        matches = search_products(deployment_type=DeploymentType.FIXED_SITE)
        self.assertEqual(
            {product.model for product in matches},
            {"SBE 16plus V2 SeaCAT", "SBE 37 MicroCAT"},
        )

    def test_unknown_depth_is_not_treated_as_supported(self) -> None:
        matches = search_products(minimum_depth_m=100)
        self.assertNotIn("SBE 37 MicroCAT", {item.model for item in matches})

    def test_no_result_does_not_invent_product(self) -> None:
        self.assertEqual(search_products(minimum_depth_m=11000), [])

    def test_get_product_spec_is_case_insensitive(self) -> None:
        product = get_product_spec("SEABIRD-SBE-19PLUS-V2")
        self.assertIsNotNone(product)
        self.assertEqual(product.model, "SBE 19plus V2 SeaCAT")

    def test_unknown_model_returns_none(self) -> None:
        self.assertIsNone(get_product_spec("不存在的型号"))

    def test_negative_depth_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "不能小于 0"):
            search_products(minimum_depth_m=-1)


if __name__ == "__main__":
    unittest.main()
