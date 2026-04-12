import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "kb.faiss"
META_PATH = DATA_DIR / "kb_meta.json"

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_meta(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_index(path: Path) -> faiss.Index:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    return faiss.read_index(str(path))


def simple_router(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Простой роутер для MVP.
    Возвращает (category, chunk_type).
    """
    q = query.lower()

    category = None
    chunk_type = None

    # Категория
    if any(x in q for x in ["товар", "магазин", "продав", "чек", "дефект", "брак"]):
        category = "return_goods"
    elif any(x in q for x in ["жкх", "ук", "тсж", "отоплен", "квартплат", "вода", "гжи", "дом"]):
        category = "housing_utilities"

    # Тип чанка
    if "иск" in q or "исков" in q:
        chunk_type = "lawsuit_template"
    elif "претензи" in q:
        chunk_type = "pretrial_claim_template"
    elif "доказатель" in q:
        chunk_type = "evidence"
    elif "что делать" in q or "алгоритм" in q or "шаг" in q:
        chunk_type = "algorithm"
    elif "закон" in q or "норматив" in q or "статья" in q:
        chunk_type = "laws"
    elif "различи" in q:
        chunk_type = "differences"
    elif "нельзя" in q or "не должна" in q:
        chunk_type = "forbidden_behavior"

    return category, chunk_type


class Retriever:
    def __init__(self, index: faiss.Index, meta: List[Dict], model_name: str):
        self.index = index
        self.meta = meta
        self.model = SentenceTransformer(model_name)

    def embed_query(self, query: str) -> np.ndarray:
        vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        return vector

    def _filter_candidates(
        self,
        category: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> List[int]:
        ids = []
        for i, chunk in enumerate(self.meta):
            ok_category = category is None or chunk["category"] == category or chunk["category"] == "meta"
            ok_type = chunk_type is None or chunk["chunk_type"] == chunk_type or chunk["category"] == "meta"

            if ok_category and ok_type:
                ids.append(i)
        return ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> List[Dict]:
        candidate_ids = self._filter_candidates(category=category, chunk_type=chunk_type)
        if not candidate_ids:
            return []

        # Берём подматрицу кандидатов
        dim = self.index.d
        candidate_matrix = np.empty((len(candidate_ids), dim), dtype="float32")
        for row_idx, chunk_idx in enumerate(candidate_ids):
            candidate_matrix[row_idx] = self.index.reconstruct(chunk_idx)

        sub_index = faiss.IndexFlatIP(dim)
        sub_index.add(candidate_matrix)

        q = self.embed_query(query)
        scores, local_ids = sub_index.search(q, min(top_k, len(candidate_ids)))

        results = []
        for score, local_id in zip(scores[0], local_ids[0]):
            global_id = candidate_ids[local_id]
            chunk = self.meta[global_id]
            results.append(
                {
                    "score": float(score),
                    "id": chunk["id"],
                    "section": chunk["section"],
                    "category": chunk["category"],
                    "chunk_type": chunk["chunk_type"],
                    "title": chunk["title"],
                    "text": chunk["text"],
                }
            )
        return results


def print_results(query: str, results: List[Dict]) -> None:
    print("=" * 100)
    print(f"Запрос: {query}")
    print("=" * 100)

    if not results:
        print("Ничего не найдено.")
        return

    for i, r in enumerate(results, start=1):
        print(f"\n[{i}] score={r['score']:.4f}")
        print(f"id        : {r['id']}")
        print(f"section   : {r['section']}")
        print(f"category  : {r['category']}")
        print(f"chunk_type: {r['chunk_type']}")
        print(f"title     : {r['title']}")
        print("-" * 100)
        preview = r["text"][:1000].strip()
        print(preview)
        if len(r["text"]) > 1000:
            print("\n...[обрезано]...")


def main() -> None:
    index = load_index(INDEX_PATH)
    meta = load_meta(META_PATH)
    retriever = Retriever(index=index, meta=meta, model_name=EMBED_MODEL_NAME)

    examples = [
        "какие доказательства нужны для возврата испорченного товара",
        "составь претензию в УК из-за холодных батарей",
        "что делать при завышенной квартплате",
        "нужен иск к продавцу за бракованный товар",
        "какие правила должна соблюдать модель при генерации документов",
    ]

    for query in examples:
        category, chunk_type = simple_router(query)
        results = retriever.search(
            query=query,
            top_k=4,
            category=category,
            chunk_type=chunk_type,
        )
        print_results(query, results)


if __name__ == "__main__":
    main()