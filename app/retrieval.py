import csv
import json
from pathlib import Path
from typing import Protocol

import numpy as np


class VectorStore(Protocol):
    def search(self, query: np.ndarray, mode: str, top_k: int) -> list[dict[str, float | str]]:
        ...


class CsvVectorStore:
    def __init__(self, csv_path: Path, model_version: str = "finetuned") -> None:
        embedding_columns = {
            "label": f"embedding_label_{model_version}",
            "full": f"embedding_full_{model_version}",
        }
        rows_by_mode: dict[str, list[tuple[str, np.ndarray]]] = {"label": [], "full": []}

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
                for mode, column in embedding_columns.items():
                    try:
                        vector = np.asarray(json.loads(row[column]), dtype=np.float32)
                    except (TypeError, json.JSONDecodeError) as error:
                        raise ValueError(f"Некорректный embedding в строке {row_number}, колонка {column}") from error
                    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                        raise ValueError(f"Некорректный вектор в строке {row_number}, колонка {column}")
                    rows_by_mode[mode].append((classname, vector))

        self._entries: dict[str, tuple[list[str], np.ndarray]] = {}
        for mode, entries in rows_by_mode.items():
            if not entries:
                raise ValueError(f"В CSV нет эталонных векторов для режима {mode}")
            dimensions = {vector.size for _, vector in entries}
            if len(dimensions) != 1:
                raise ValueError(f"Размерность векторов для режима {mode} различается")
            classes = [classname for classname, _ in entries]
            matrix = np.stack([vector for _, vector in entries]).astype(np.float32, copy=False)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError(f"CSV содержит нулевой вектор в режиме {mode}")
            self._entries[mode] = (classes, matrix / norms)

    def search(self, query: np.ndarray, mode: str, top_k: int) -> list[dict[str, float | str]]:
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

        scores_by_class: dict[str, float] = {}
        for classname, score in zip(classes, matrix @ (query / norm)):
            scores_by_class[classname] = max(scores_by_class.get(classname, -np.inf), float(score))
        ranked = sorted(scores_by_class.items(), key=lambda item: item[1], reverse=True)
        return [
            {"classname": classname, "similarity": score}
            for classname, score in ranked[: max(1, top_k)]
        ]


class PgVectorStore:
    def __init__(self, dsn: str, model_version: str = "finetuned") -> None:
        if not dsn:
            raise ValueError("Для VECTOR_BACKEND=pgvector необходимо задать PG_DSN")
        self.dsn = dsn
        self.model_version = model_version

    def search(self, query: np.ndarray, mode: str, top_k: int) -> list[dict[str, float | str]]:
        try:
            import psycopg
        except ImportError as error:
            raise RuntimeError("Для Pgvector установите зависимости из requirements.txt") from error

        vector_literal = "[" + ",".join(str(float(value)) for value in query) + "]"
        image_type = "crop" if mode == "label" else "full"
        sql = """
            SELECT p.wine_name, MAX(1 - (pe.embedding <=> %s::vector)) AS similarity
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
        return [{"classname": row[0], "similarity": float(row[1])} for row in rows]


def create_vector_store(backend: str, csv_path: Path, pg_dsn: str) -> VectorStore:
    if backend == "csv":
        return CsvVectorStore(csv_path)
    if backend == "pgvector":
        return PgVectorStore(pg_dsn)
    raise ValueError("VECTOR_BACKEND должен быть 'csv' или 'pgvector'")
