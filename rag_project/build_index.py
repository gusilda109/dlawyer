import json
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
CHUNKS_PATH = DATA_DIR / "kb_chunks.json"
INDEX_PATH = DATA_DIR / "kb.faiss"
META_PATH = DATA_DIR / "kb_meta.json"

# Для русского MVP без лишней тяжести
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_chunks(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_corpus(chunks: List[Dict]) -> List[str]:
    """
    Текст для эмбеддинга.
    Добавляем title и служебные поля в текст,
    чтобы retrieval лучше чувствовал тип чанка.
    """
    corpus = []
    for chunk in chunks:
        text = (
            f"category: {chunk['category']}\n"
            f"chunk_type: {chunk['chunk_type']}\n"
            f"title: {chunk['title']}\n\n"
            f"{chunk['text']}"
        )
        corpus.append(text)
    return corpus


def encode_corpus(model_name: str, corpus: List[str]) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        corpus,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Используем IndexFlatIP.
    Так как эмбеддинги нормализованы, inner product работает как cosine-поиск.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def save_meta(chunks: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)
    corpus = build_corpus(chunks)
    embeddings = encode_corpus(EMBED_MODEL_NAME, corpus)
    index = build_faiss_index(embeddings)

    save_index(index, INDEX_PATH)
    save_meta(chunks, META_PATH)

    print("Готово.")
    print(f"Индекс сохранён: {INDEX_PATH}")
    print(f"Метаданные сохранены: {META_PATH}")
    print(f"Чанков в индексе: {len(chunks)}")
    print(f"Размерность эмбеддинга: {embeddings.shape[1]}")


if __name__ == "__main__":
    main()