import csv
from pathlib import Path
from typing import Any, Protocol


class ProductStore(Protocol):
    def fetch_by_names(self, names: list[str]) -> dict[str, dict[str, Any]]:
        ...

    def fetch_by_ids(self, product_ids: list[int]) -> dict[int, dict[str, Any]]:
        ...


def normalize_product_name(value: str) -> str:
    return " ".join(value.casefold().split())


def _merge_product_matches(matches: list[dict[str, Any]]) -> dict[str, Any]:
    if len(matches) == 1:
        return matches[0]

    merged = dict(matches[0])
    for field in ("id", "slug", "photo_name", "created_at", "updated_at"):
        merged[field] = None

    for field in ("category", "color", "region", "grape_variety", "description", "winery"):
        values = list(
            dict.fromkeys(
                str(product[field]).strip()
                for product in matches
                if product.get(field) is not None and str(product[field]).strip()
            )
        )
        if field == "description" and len(values) > 1:
            merged[field] = None
        else:
            merged[field] = " / ".join(values) or None
    return merged


class CsvProductStore:
    _columns = {
        "wine_name": "Название вина",
        "category": "Категория",
        "color": "Цвет",
        "region": "Регион",
        "grape_variety": "Сорт винограда",
        "description": "Описание",
        "winery": "Винодельня",
        "slug": "Slug",
        "photo_name": "Название фото",
    }

    def __init__(self, csv_path: Path) -> None:
        self._products: dict[str, list[dict[str, Any]]] = {}
        self._products_by_id: dict[int, dict[str, Any]] = {}
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            missing = set(self._columns.values()).difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"В CSV товаров отсутствуют колонки: {', '.join(sorted(missing))}")
            for product_id, row in enumerate(reader, start=1):
                wine_name = (row.get(self._columns["wine_name"]) or "").strip()
                slug = (row.get(self._columns["slug"]) or "").strip()
                if not wine_name or not slug:
                    raise ValueError(f"В CSV товаров есть пустое название или slug в строке {product_id + 1}")
                product = {
                    "id": product_id,
                    **{
                        field: (row.get(source_name) or "").strip() or None
                        for field, source_name in self._columns.items()
                    },
                    "created_at": None,
                    "updated_at": None,
                }
                self._products.setdefault(normalize_product_name(wine_name), []).append(product)
                self._products_by_id[product_id] = product

    def fetch_by_names(self, names: list[str]) -> dict[str, dict[str, Any]]:
        products: dict[str, dict[str, Any]] = {}
        for name in names:
            key = normalize_product_name(name)
            matches = self._products.get(key, [])
            if matches:
                products[key] = _merge_product_matches(matches)
        return products

    def fetch_by_ids(self, product_ids: list[int]) -> dict[int, dict[str, Any]]:
        return {
            product_id: self._products_by_id[product_id]
            for product_id in product_ids
            if product_id in self._products_by_id
        }


class PgProductStore:
    def __init__(self, dsn: str) -> None:
        if not dsn:
            raise ValueError("Для получения карточек товаров необходимо задать PG_DSN")
        self.dsn = dsn

    def fetch_by_names(self, names: list[str]) -> dict[str, dict[str, Any]]:
        if not names:
            return {}
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError("Для PostgreSQL установите зависимости из requirements.txt") from error

        sql = """
            SELECT id, wine_name, category, color, region, grape_variety,
                   description, winery, slug, photo_name, created_at, updated_at
            FROM products
            WHERE lower(btrim(wine_name)) = ANY(%s)
        """
        normalized_names = [normalize_product_name(name) for name in names]
        with psycopg.connect(self.dsn) as connection:
            rows = connection.execute(sql, (normalized_names,)).fetchall()
        columns = (
            "id", "wine_name", "category", "color", "region", "grape_variety",
            "description", "winery", "slug", "photo_name", "created_at", "updated_at",
        )
        matches_by_name: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            key = normalize_product_name(str(row[1]))
            matches_by_name.setdefault(key, []).append(dict(zip(columns, row)))
        return {
            key: _merge_product_matches(matches)
            for key, matches in matches_by_name.items()
        }

    def fetch_by_ids(self, product_ids: list[int]) -> dict[int, dict[str, Any]]:
        if not product_ids:
            return {}
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError("Для PostgreSQL установите зависимости из requirements.txt") from error

        sql = """
            SELECT id, wine_name, category, color, region, grape_variety,
                   description, winery, slug, photo_name, created_at, updated_at
            FROM products
            WHERE id = ANY(%s)
        """
        with psycopg.connect(self.dsn) as connection:
            rows = connection.execute(sql, (product_ids,)).fetchall()
        columns = (
            "id", "wine_name", "category", "color", "region", "grape_variety",
            "description", "winery", "slug", "photo_name", "created_at", "updated_at",
        )
        return {int(row[0]): dict(zip(columns, row)) for row in rows}


def enrich_predictions(
    predictions: list[dict[str, float | int | str]],
    products_by_name: dict[str, dict[str, Any]],
    products_by_id: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    products_by_id = products_by_id or {}
    resolved: list[tuple[dict[str, float | int | str], dict[str, Any]]] = []
    missing: list[str] = []
    for prediction in predictions:
        classname = str(prediction["classname"])
        if "product_id" in prediction:
            try:
                product = products_by_id.get(int(prediction["product_id"]))
            except (TypeError, ValueError):
                product = None
        else:
            product = products_by_name.get(normalize_product_name(classname))
        if product is None:
            missing.append(classname)
        else:
            resolved.append((prediction, product))
    if missing:
        raise ValueError(f"В таблице products не найдены товары: {', '.join(missing)}")

    return [
        {
            "classname": prediction["classname"],
            "similarity": prediction["similarity"],
            "product": product,
        }
        for prediction, product in resolved
    ]
