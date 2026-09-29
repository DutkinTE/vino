import csv
import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

PRODUCT_COLUMNS = {
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


def main() -> None:
    try:
        import psycopg
    except ImportError as error:
        raise SystemExit("Установите зависимости: pip install -r requirements.txt") from error

    dsn = os.getenv("PG_DSN", "")
    if not dsn:
        raise SystemExit("Задайте PG_DSN в .env")
    csv_path = Path(os.getenv("PRODUCTS_CSV", "data/reference/product_info.csv"))
    if not csv_path.is_absolute():
        csv_path = ROOT_DIR / csv_path
    if not csv_path.is_file():
        raise SystemExit(f"Не найден CSV товаров: {csv_path}")

    columns = tuple(PRODUCT_COLUMNS)
    placeholders = ", ".join(["%s"] * len(columns))
    updates = ", ".join(
        f"{column} = EXCLUDED.{column}" for column in columns if column != "slug"
    )
    insert_sql = f"""
        INSERT INTO products ({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT (slug) DO UPDATE SET {updates}
    """

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = set(PRODUCT_COLUMNS.values()).difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"В CSV товаров отсутствуют колонки: {', '.join(sorted(missing))}")
        records = []
        slugs = set()
        for row_number, row in enumerate(reader, start=2):
            values = [(row.get(source_name) or "").strip() or None for source_name in PRODUCT_COLUMNS.values()]
            wine_name = values[0]
            slug = values[7]
            if not wine_name:
                raise SystemExit(f"Пустое название вина в строке {row_number}")
            if not slug:
                raise SystemExit(f"Пустой slug в строке {row_number}")
            if slug in slugs:
                raise SystemExit(f"Повторяющийся slug '{slug}' в CSV (строка {row_number})")
            slugs.add(slug)
            records.append(values)

    count = 0
    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            for values in records:
                cursor.execute(insert_sql, values)
                count += 1
    print(f"Загружено/обновлено карточек товаров: {count}")


if __name__ == "__main__":
    main()