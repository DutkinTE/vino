import unittest
import json
import csv
import tempfile
from pathlib import Path

from app.products import CsvProductStore, enrich_predictions
from scripts.import_csv_to_pgvector import find_product_id_by_name, parse_embedding_rows


class ProductEnrichmentTests(unittest.TestCase):
    def test_attaches_product_details_without_changing_match_order(self) -> None:
        matches = [
            {"classname": "rose", "similarity": 0.97},
            {"classname": "white", "similarity": 0.83},
        ]
        products = {
            "rose": {"wine_name": "rose", "region": "North"},
            "white": {"wine_name": "white", "region": "South"},
        }

        enriched = enrich_predictions(matches, products)

        self.assertEqual([item["classname"] for item in enriched], ["rose", "white"])
        self.assertEqual(enriched[0]["similarity"], 0.97)
        self.assertEqual(enriched[0]["product"]["region"], "North")
        self.assertEqual(enriched[1]["product"]["region"], "South")

    def test_reports_missing_product_records(self) -> None:
        with self.assertRaisesRegex(ValueError, "white"):
            enrich_predictions([{"classname": "white", "similarity": 0.8}], {})

    def test_matches_product_names_without_case_sensitivity(self) -> None:
        enriched = enrich_predictions(
            [{"classname": "ROSE", "similarity": 0.8}],
            {"rose": {"wine_name": "Rose"}},
        )

        self.assertEqual(enriched[0]["product"]["wine_name"], "Rose")


class CsvProductStoreTests(unittest.TestCase):
    def test_loads_product_details_without_postgres(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "products.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "Название вина", "Категория", "Цвет", "Регион",
                        "Сорт винограда", "Описание", "Винодельня", "Slug", "Название фото",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "Название вина": "Rose Wine",
                    "Категория": "wine",
                    "Цвет": "rose",
                    "Регион": "Италия",
                    "Сорт винограда": "Sangiovese",
                    "Описание": "Сухое",
                    "Винодельня": "Winery",
                    "Slug": "rose-wine",
                    "Название фото": "rose.webp",
                })

            products = CsvProductStore(csv_path).fetch_by_names(["  ROSE   WINE "])

        self.assertEqual(products["rose wine"]["wine_name"], "Rose Wine")
        self.assertEqual(products["rose wine"]["id"], 1)


class EmbeddingImportTests(unittest.TestCase):
    def test_parses_all_four_embedding_columns(self) -> None:
        row = {
            "wine_name": "Rose Wine",
            "embedding_label_base": json.dumps([0.1] * 768),
            "embedding_full_base": json.dumps([0.2] * 768),
            "embedding_label_finetuned": json.dumps([0.3] * 768),
            "embedding_full_finetuned": json.dumps([0.4] * 768),
        }

        embeddings = parse_embedding_rows([row])

        self.assertEqual(len(embeddings), 4)
        self.assertEqual(
            [(item[2], item[3]) for item in embeddings],
            [
                ("crop", "base"),
                ("full", "base"),
                ("crop", "finetuned"),
                ("full", "finetuned"),
            ],
        )
        self.assertEqual(embeddings[0][4], "[" + ",".join(["0.1"] * 768) + "]")

    def test_matches_product_by_normalized_name_without_filename(self) -> None:
        product_id, reason = find_product_id_by_name(
            "  ROSE   WINE ",
            {"rose wine": [7]},
        )

        self.assertEqual(product_id, 7)
        self.assertIsNone(reason)

    def test_rejects_ambiguous_and_missing_product_names(self) -> None:
        ambiguous_id, ambiguous_reason = find_product_id_by_name(
            "Pinot Noir",
            {"pinot noir": [1, 2]},
        )
        missing_id, missing_reason = find_product_id_by_name("Other wine", {})

        self.assertIsNone(ambiguous_id)
        self.assertEqual(ambiguous_reason, "ambiguous")
        self.assertIsNone(missing_id)
        self.assertEqual(missing_reason, "not_found")


if __name__ == "__main__":
    unittest.main()
