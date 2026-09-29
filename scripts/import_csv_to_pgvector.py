import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
UNMATCHED_REPORT = ROOT_DIR / "data/reference/unmatched_product_embeddings.csv"
MIN_MATCH_SCORE = 0.6
MIN_SCORE_MARGIN = 0.15
EMBEDDING_SPECS = (
    ("base", "label", "crop"),
    ("base", "full", "full"),
    ("finetuned", "label", "crop"),
    ("finetuned", "full", "full"),
)


def parse_embedding_rows(
    rows: Iterable[dict[str, str | None]],
    name_column: str = "wine_name",
    product_id_column: str = "product_id",
) -> list[tuple[str, str, str, str, str, int | None]]:
    embeddings = []
    for row_number, row in enumerate(rows, start=2):
        product_name = (row.get(name_column) or "").strip()
        if not product_name:
            raise ValueError(f"Пустое название товара в строке {row_number}, колонка {name_column}")
        raw_product_id = (row.get(product_id_column) or "").strip()
        try:
            product_id = int(raw_product_id) if raw_product_id else None
        except ValueError as error:
            raise ValueError(
                f"Некорректный product_id в строке {row_number}, колонка {product_id_column}"
            ) from error
        if product_id is not None and product_id < 1:
            raise ValueError(f"product_id в строке {row_number} должен быть положительным числом")
        filename = (row.get("filename") or "").strip()
        for model_version, mode, image_type in EMBEDDING_SPECS:
            column = f"embedding_{mode}_{model_version}"
            try:
                vector = json.loads(row.get(column) or "")
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Некорректный embedding в строке {row_number}, колонка {column}"
                ) from error
            if (
                not isinstance(vector, list)
                or len(vector) != 768
                or any(
                    not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in vector
                )
            ):
                raise ValueError(
                    f"Embedding в строке {row_number}, колонка {column} "
                    "должен быть конечным вектором размерности 768"
                )
            vector_literal = "[" + ",".join(str(float(value)) for value in vector) + "]"
            embeddings.append((product_name, filename, image_type, model_version, vector_literal, product_id))
    return embeddings


def normalize_product_name(value: str) -> str:
    return " ".join(value.casefold().split())


def find_product_id_by_name(
    product_name: str,
    products_by_name: dict[str, list[int]],
) -> tuple[int | None, str | None]:
    product_ids = products_by_name.get(normalize_product_name(product_name), [])
    if not product_ids:
        return None, "not_found"
    if len(product_ids) > 1:
        return None, "ambiguous"
    return product_ids[0], None


def main() -> None:
    try:
        import psycopg
    except ImportError as error:
        raise SystemExit("Установите зависимости: pip install -r requirements.txt") from error

    dsn = os.getenv("PG_DSN", "")
    if not dsn:
        raise SystemExit("Задайте PG_DSN в .env")
    csv_path = Path(os.getenv("REFERENCE_CSV", "data/reference/embedding_comparison_reference.csv"))
    if not csv_path.is_absolute():
        csv_path = ROOT_DIR / csv_path
    if not csv_path.is_file():
        raise SystemExit(f"Не найден CSV: {csv_path}")

    insert_sql = """
        INSERT INTO product_embeddings
            (product_id, filename, image_type, model_version, embedding)
        VALUES (%s, %s, %s, %s, %s::vector)
        ON CONFLICT (product_id, image_type, model_version)
        DO UPDATE SET filename = EXCLUDED.filename, embedding = EXCLUDED.embedding
    """

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        name_column = "wine_name" if "wine_name" in fieldnames else "classname"
        required = {name_column, *(f"embedding_{mode}_{version}" for version, mode, _ in EMBEDDING_SPECS)}
        missing = required.difference(fieldnames)
        if missing:
            raise SystemExit(f"В CSV эмбеддингов отсутствуют колонки: {', '.join(sorted(missing))}")
        try:
            embeddings = parse_embedding_rows(reader, name_column)
        except ValueError as error:
            raise SystemExit(str(error)) from error

    count = 0
    unmatched = []
    with psycopg.connect(dsn) as connection:
        product_rows = connection.execute(
            "SELECT id, wine_name FROM products"
        ).fetchall()
        product_ids = {int(product_id) for product_id, _ in product_rows}
        products_by_name: dict[str, list[int]] = {}
        for product_id, wine_name in product_rows:
            key = normalize_product_name(str(wine_name))
            products_by_name.setdefault(key, []).append(int(product_id))

        with connection.cursor() as cursor:
            for product_name, filename, image_type, model_version, vector_literal, explicit_product_id in embeddings:
                if explicit_product_id is not None:
                    product_id = explicit_product_id if explicit_product_id in product_ids else None
                    reason = None if product_id is not None else "not_found"
                else:
                    product_id, reason = find_product_id_by_name(product_name, products_by_name)
                if product_id is None:
                    unmatched.append(
                        (product_name, filename, image_type, model_version, reason or "unmatched")
                    )
                    continue
                cursor.execute(
                    insert_sql,
                    (product_id, filename, image_type, model_version, vector_literal),
                )
                count += 1

    if unmatched:
        with UNMATCHED_REPORT.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ("wine_name", "filename", "image_type", "model_version", "reason")
            )
            writer.writerows(unmatched)
    elif UNMATCHED_REPORT.exists():
        UNMATCHED_REPORT.unlink()

    print(f"Загружено/обновлено векторов: {count}")
    print(f"Пропущено векторов из-за отсутствующей или неоднозначной карточки: {len(unmatched)}")
    if unmatched:
        print(f"Список пропусков: {UNMATCHED_REPORT}")


if __name__ == "__main__":
    main()
