import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ocean_agent.build_document_chunks import build_document_chunks
from ocean_agent.document_cleaner import (
    clean_page_text,
    find_repeated_margin_lines,
    infer_chunk_type,
    is_table_of_contents,
    looks_like_pdf_heading,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DocumentCleanerTests(unittest.TestCase):
    def test_repairs_common_pdf_text_artifacts(self) -> None:
        cleaned, actions = clean_page_text(
            "The sensor minimiz-\nes noise at 4. 5V.\nO- ring maintenance\n"
            "seabird.\ncom\n•\n\uf16f"
        )

        self.assertIn("minimizes", cleaned)
        self.assertIn("4.5V", cleaned)
        self.assertIn("O-ring", cleaned)
        self.assertIn("seabird.com", cleaned)
        self.assertIn("1. 2. 3.", clean_page_text("1. 2. 3. Steps")[0])
        self.assertIn("joined_hyphenated_words", actions)
        self.assertNotIn("\uf16f", cleaned)
        self.assertNotIn("•", cleaned)

    def test_removes_repeated_margin_and_standalone_page_number(self) -> None:
        pages = tuple(
            f"RBR Instrument Guide\nUseful page {number} content.\nRBR#0013236revF {number}"
            for number in range(1, 7)
        )
        repeated = find_repeated_margin_lines(pages)
        cleaned, actions = clean_page_text(
            pages[0], repeated_margin_lines=repeated
        )

        self.assertNotIn("RBR Instrument Guide", cleaned)
        self.assertNotIn("RBR#0013236revF", cleaned)
        self.assertIn("Useful page 1 content", cleaned)
        self.assertIn("removed_repeated_margin", actions)

    def test_detects_table_of_contents(self) -> None:
        text = (
            "Table of contents\n"
            "1 Introduction ................. 4\n"
            "2 Hardware ..................... 9\n"
            "3 Maintenance .................. 25"
        )
        self.assertTrue(is_table_of_contents(text))

    def test_marks_scrambled_procedure_and_pinout_for_review(self) -> None:
        self.assertEqual(
            infer_chunk_type("1. 2. 3. 4. Replacing the O-ring")[1],
            "needs_review",
        )
        self.assertEqual(
            infer_chunk_type("Pin No. USB RS-232 RS-485 Ground")[1],
            "needs_review",
        )

    def test_heading_detection_rejects_table_rows_and_part_numbers(self) -> None:
        self.assertTrue(looks_like_pdf_heading("5.2 Instrument interface"))
        self.assertTrue(looks_like_pdf_heading("5 Hardware"))
        self.assertFalse(looks_like_pdf_heading("1 Ground"))
        self.assertFalse(looks_like_pdf_heading("3 N/C From the"))
        self.assertFalse(looks_like_pdf_heading("50435 Spares kit with cable"))


class DocumentChunkBuildTests(unittest.TestCase):
    def test_builds_inspectable_json_with_hashes_and_neighbors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            output = root / "document_chunks.json"
            cleaned_dir = root / "cleaned"
            payload = build_document_chunks(
                PROJECT_ROOT / "documents",
                manifest_names=("manifest.json",),
                output_path=output,
                cleaned_output_dir=cleaned_dir,
                max_chars=100,
            )

            saved = json.loads(output.read_text(encoding="utf-8"))
            chunks = saved["chunks"]
            self.assertEqual(saved["summary"], payload["summary"])
            self.assertGreater(len(chunks), 1)
            self.assertIsNone(chunks[0]["previous_chunk_id"])
            self.assertEqual(chunks[0]["next_chunk_id"], chunks[1]["chunk_id"])
            self.assertEqual(
                chunks[0]["content_hash"],
                hashlib.sha256(chunks[0]["content"].encode()).hexdigest(),
            )
            self.assertTrue(
                (cleaned_dir / "sbe19plus-v2-project-summary.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
