import json
import re
from pathlib import Path
from typing import List, Dict
from docx import Document

DATA_DIR = Path("data")
DOCX_PATH = DATA_DIR / "rag_knowledge_base.docx"
CHUNKS_PATH = DATA_DIR / "kb_chunks.json"


def load_docx_text(path: Path) -> str:
    """Читает docx и возвращает весь непустой текст одним блоком."""
    doc = Document(path)
    parts: List[str] = []

    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            parts.append(text)

    return "\n".join(parts)


def infer_category(section_num: str) -> str:
    """Определяет верхнеуровневую категорию по номеру раздела."""
    if section_num.startswith("1."):
        return "return_goods"
    if section_num.startswith("2."):
        return "housing_utilities"
    if section_num.startswith("3."):
        return "meta"
    return "unknown"


def infer_chunk_type(title: str) -> str:
    """Определяет тип чанка по названию раздела."""
    t = title.lower()

    if "норматив" in t:
        return "laws"
    if "классификация требований" in t:
        return "claims_classification"
    if "типичные нарушения" in t:
        return "violations"
    if "алгоритм" in t:
        return "algorithm"
    if "перечень доказательств" in t or "доказательств" in t:
        return "evidence"
    if "досудебной претензии" in t or "претензии" in t:
        return "pretrial_claim_template"
    if "искового заявления" in t or "иска" in t:
        return "lawsuit_template"
    if "правила использования шаблонов" in t:
        return "usage_rules"
    if "ключевые различия" in t:
        return "differences"
    if "что модель не должна делать" in t:
        return "forbidden_behavior"

    return "other"


def extract_sections(text: str) -> List[Dict]:
    """
    Ищет заголовки формата:
    1.1. Нормативная база
    2.5. Шаблон досудебной претензии в УК/ТСЖ
    3.1. Правила использования шаблонов
    """
    pattern = re.compile(r"(?m)^((?:\d+\.\d+))\.\s+(.+)$")
    matches = list(pattern.finditer(text))

    if not matches:
        raise ValueError("Не удалось найти заголовки разделов формата '1.1. Название'.")

    chunks: List[Dict] = []

    for i, match in enumerate(matches):
        section_num = match.group(1).strip()
        title = match.group(2).strip()

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        category = infer_category(section_num)
        chunk_type = infer_chunk_type(title)

        chunks.append(
            {
                "id": f"{section_num}_{chunk_type}",
                "section": section_num,
                "category": category,
                "chunk_type": chunk_type,
                "title": title,
                "text": body,
            }
        )

    return chunks


def save_chunks(chunks: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def main() -> None:
    if not DOCX_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {DOCX_PATH}")

    text = load_docx_text(DOCX_PATH)
    chunks = extract_sections(text)
    save_chunks(chunks, CHUNKS_PATH)

    print(f"Готово. Сохранено чанков: {len(chunks)}")
    print(f"Файл: {CHUNKS_PATH}")

    for chunk in chunks:
        print(f"- {chunk['section']} | {chunk['category']} | {chunk['chunk_type']} | {chunk['title']}")


if __name__ == "__main__":
    main()