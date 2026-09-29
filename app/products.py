import csv
from pathlib import Path
from typing import Any, Protocol


class ProductStore(Protocol):
    def fetch_by_names(self, names: list[str]) -> dict[str, dict[str, Any]]:
        ...


def normalize_product_name(value: str) -> str:
    return " ".join(value.casefold().split())


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

    def fetch_by_names(self, names: list[str]) -> dict[str, dict[str, Any]]:
        products: dict[str, dict[str, Any]] = {}
        for name in names:
            key = normalize_product_name(name)
            matches = self._products.get(key, [])
            if len(matches) > 1:
                raise ValueError(f"Неоднозначное название товара в products: {name}")
            if matches:
                products[key] = matches[0]
        return products


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
        products: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = normalize_product_name(str(row[1]))
            if key in products:
                raise ValueError(f"Неоднозначное название товара в products: {row[1]}")
            products[key] = dict(zip(columns, row))
        return products


def enrich_predictions(
    predictions: list[dict[str, float | str]],
    products_by_name: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    missing = [
        str(prediction["classname"])
        for prediction in predictions
        if normalize_product_name(str(prediction["classname"])) not in products_by_name
    ]
    if missing:
        raise ValueError(f"В таблице products не найдены товары: {', '.join(missing)}")

    return [
        {
            "classname": prediction["classname"],
            "similarity": prediction["similarity"],
            "product": products_by_name[normalize_product_name(str(prediction["classname"]))],
        }
        for prediction in predictions
    ]
