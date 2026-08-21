import tempfile
import unittest
from pathlib import Path

from ocean_agent.sessions import build_session


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_session_id_reopens_saved_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "conversations.db"
            first = build_session("learning-demo", db_path)
            await first.add_items(
                [{"role": "user", "content": "我需要一个5000米CTD"}]
            )
            first.close()

            reopened = build_session("learning-demo", db_path)
            items = await reopened.get_items()
            reopened.close()

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["role"], "user")
        self.assertEqual(items[0]["content"], "我需要一个5000米CTD")

    async def test_different_session_ids_have_separate_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "conversations.db"
            first = build_session("customer-a", db_path)
            await first.add_items([{"role": "user", "content": "A的需求"}])
            first.close()

            second = build_session("customer-b", db_path)
            items = await second.get_items()
            second.close()

        self.assertEqual(items, [])

    def test_empty_session_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "session_id 不能为空"):
            build_session("   ")


if __name__ == "__main__":
    unittest.main()
