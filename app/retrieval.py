import csv
import json
from pathlib import Path
from typing import Protocol

import numpy as np


class VectorStore(Protocol):
    def search(self, query: np.ndarray, mode: str, top_k: int) -> list[dict[str, float | int | str]]:
        ...


class CsvVectorStore:
    def __init__(self, csv_path: Path, model_version: str = "finetuned") -> None:
        embedding_columns = {
            "label": f"embedding_label_{model_version}",
            "full": f"embedding_full_{model_version}",
        }
        rows_by_mode: dict[str, list[tuple[str, int | None, np.ndarray]]] = {"label": [], "full": []}

        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = set(reader.fieldnames or [])
            classname_column = "classname" if "classname" in fieldnames else "wine_name"
            required = {classname_column, *embedding_columns.values()}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"В CSV отсутствуют колонки: {', '.join(sorted(missing))}")

            for row_number, row in enumerate(reader, start=2):
                classname = (row.get(classname_column) or "").strip()
                if not classname:
                    continue
                raw_product_id = (row.get("product_id") or "").strip()
                try:
                    product_id = int(raw_product_id) if raw_product_id else None
                except ValueError as error:
                    raise ValueError(f"Некорректный product_id в строке {row_number}") from error
                for mode, column in embedding_columns.items():
                    try:
                        vector = np.asarray(json.loads(row[column]), dtype=np.float32)
                    except (TypeError, json.JSONDecodeError) as error:
                        raise ValueError(f"Некорректный embedding в строке {row_number}, колонка {column}") from error
                    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                        raise ValueError(f"Некорректный вектор в строке {row_number}, колонка {column}")
                    rows_by_mode[mode].append((classname, product_id, vector))

        self._entries: dict[str, tuple[list[tuple[str, int | None]], np.ndarray]] = {}
        for mode, entries in rows_by_mode.items():
            if not entries:
                raise ValueError(f"В CSV нет эталонных векторов для режима {mode}")
            dimensions = {vector.size for _, _, vector in entries}
            if len(dimensions) != 1:
                raise ValueError(f"Размерность векторов для режима {mode} различается")
            classes = [(classname, product_id) for classname, product_id, _ in entries]
            matrix = np.stack([vector for _, _, vector in entries]).astype(np.float32, copy=False)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError(f"CSV содержит нулевой вектор в режиме {mode}")
            self._entries[mode] = (classes, matrix / norms)

    def search(self, query: np.ndarray, mode: str, top_k: int) -> list[dict[str, float | int | str]]:
        if mode not in self._entries:
            raise ValueError(f"Неизвестный режим поиска: {mode}")
        classes, matrix = self._entries[mode]
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.size != matrix.shape[1]:
            raise ValueError(
                f"Размерность query ({query.size}) не совпадает с эталонами ({matrix.shape[1]})"
            )
        norm = np.linalg.norm(query)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Query-вектор пустой или содержит некорректные значения")

        scores_by_product: dict[tuple[str, str | int], tuple[str, int | None, float]] = {}
        for (classname, product_id), score in zip(classes, matrix @ (query / norm)):
            key = ("id", product_id) if product_id is not None else ("name", classname)
            previous = scores_by_product.get(key)
            if previous is None or float(score) > previous[2]:
                scores_by_product[key] = (classname, product_id, float(score))
        ranked = sorted(scores_by_product.values(), key=lambda item: item[2], reverse=True)
        matches: list[dict[str, float | int | str]] = []
        for classname, product_id, score in ranked[: max(1, top_k)]:
            match: dict[str, float | int | str] = {"classname": classname, "similarity": score}
            if product_id is not None:
                match["product_id"] = product_id
            matches.append(match)
        return matches


class PgVectorStore:
    def __init__(self, dsn: str, model_version: str = "finetuned") -> None:
        if not dsn:
            raise ValueError("Для VECTOR_BACKEND=pgvector необходимо задать PG_DSN")
        self.dsn = dsn
        self.model_version = model_version

    def search(self, query: np.ndarray, mode: str, top_k: int) -> list[dict[str, float | int | str]]:
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError("Для Pgvector установите зависимости из requirements.txt") from error

        vector_literal = "[" + ",".join(str(float(value)) for value in query) + "]"
        image_type = "crop" if mode == "label" else "full"
        sql = """
            SELECT p.id, p.wine_name, MAX(1 - (pe.embedding <=> %s::vector)) AS similarity
            FROM product_embeddings AS pe
            JOIN products AS p ON p.id = pe.product_id
            WHERE pe.model_version = %s AND pe.image_type = %s
            GROUP BY p.id, p.wine_name
            ORDER BY similarity DESC
            LIMIT %s
        """
        with psycopg.connect(self.dsn) as connection:
            rows = connection.execute(
                sql, (vector_literal, self.model_version, image_type, max(1, top_k))
            ).fetchall()
        return [
            {"product_id": int(row[0]), "classname": row[1], "similarity": float(row[2])}
            for row in rows
        ]


def create_vector_store(backend: str, csv_path: Path, pg_dsn: str) -> VectorStore:
    if backend == "csv":
        return CsvVectorStore(csv_path)
    if backend == "pgvector":
        return PgVectorStore(pg_dsn)
    raise ValueError("VECTOR_BACKEND должен быть 'csv' или 'pgvector'")
