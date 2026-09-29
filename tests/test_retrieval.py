import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.retrieval import CsvVectorStore


class CsvVectorStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "references.csv"
        with self.csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "classname",
                    "embedding_label_finetuned",
                    "embedding_full_finetuned",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "classname": "rose",
                        "embedding_label_finetuned": json.dumps([1, 0, 0]),
                        "embedding_full_finetuned": json.dumps([0, 1, 0]),
                    },
                    {
                        "classname": "rose",
                        "embedding_label_finetuned": json.dumps([0.9, 0.1, 0]),
                        "embedding_full_finetuned": json.dumps([0, 0.9, 0.1]),
                    },
                    {
                        "classname": "white",
                        "embedding_label_finetuned": json.dumps([0, 1, 0]),
                        "embedding_full_finetuned": json.dumps([1, 0, 0]),
                    },
                ]
            )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_ranks_classes_by_cosine_and_deduplicates(self) -> None:
        store = CsvVectorStore(self.csv_path)

        predictions = store.search(np.array([1, 0, 0]), "label", top_k=5)

        self.assertEqual([item["classname"] for item in predictions], ["rose", "white"])
        self.assertAlmostEqual(predictions[0]["similarity"], 1.0)

    def test_selects_full_image_vectors_for_fallback(self) -> None:
        store = CsvVectorStore(self.csv_path)

        predictions = store.search(np.array([1, 0, 0]), "full", top_k=2)

        self.assertEqual(predictions[0]["classname"], "white")

    def test_accepts_wine_name_as_the_product_column(self) -> None:
        with self.csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "wine_name",
                    "embedding_label_finetuned",
                    "embedding_full_finetuned",
                ],
            )
            writer.writeheader()
            writer.writerow({
                "wine_name": "Rose Wine",
                "embedding_label_finetuned": json.dumps([1, 0, 0]),
                "embedding_full_finetuned": json.dumps([0, 1, 0]),
            })

        predictions = CsvVectorStore(self.csv_path).search(np.array([1, 0, 0]), "label", top_k=1)

        self.assertEqual(predictions[0]["classname"], "Rose Wine")


if __name__ == "__main__":
    unittest.main()
