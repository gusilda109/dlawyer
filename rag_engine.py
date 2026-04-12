"""
rag_engine.py — RAG-движок для «Цифрового правозащитника».

Интеграция с существующим пайплайном:
  - build_chunks.py  → data/kb_chunks.json
  - build_index.py   → data/kb.faiss + data/kb_meta.json
  - test_retrieval.py → Retriever + simple_router (переиспользуются здесь)

Два режима:
  • FAISS + SentenceTransformer  — если data/kb.faiss существует (продакшн).
  • TF-IDF fallback               — если индекс ещё не построен (первый запуск / оффлайн).

Запуск пайплайна вручную:
  python build_chunks.py   # парсит docx → data/kb_chunks.json
  python build_index.py    # строит эмбеддинги → data/kb.faiss + data/kb_meta.json
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# Пути — совпадают с build_index.py
# ─────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
CHUNKS_PATH = DATA_DIR / "kb_chunks.json"
INDEX_PATH  = DATA_DIR / "kb.faiss"
META_PATH   = DATA_DIR / "kb_meta.json"

# Та же модель, что в build_index.py
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ─────────────────────────────────────────────
# Маппинги (Streamlit UI → внутренние ключи)
# ─────────────────────────────────────────────
CATEGORY_MAP = {
    "Возврат испорченного товара": "return_goods",
    "Споры с ЖКХ":                 "housing_utilities",
}

DOC_TYPE_MAP = {
    "Досудебная претензия": "pretrial",
    "Исковое заявление":    "lawsuit",
}

# Приоритетные chunk_type по категории и типу документа
PRIORITY_CHUNKS = {
    "return_goods": {
        "pretrial": ["pretrial_claim_template", "laws", "claims_classification", "evidence"],
        "lawsuit":  ["lawsuit_template",         "laws", "claims_classification", "evidence"],
    },
    "housing_utilities": {
        "pretrial": ["pretrial_claim_template", "laws", "violations", "evidence"],
        "lawsuit":  ["lawsuit_template",         "laws", "violations", "evidence"],
    },
}

META_TYPES = {"usage_rules", "forbidden_behavior"}


# ─────────────────────────────────────────────
# simple_router — из test_retrieval.py
# ─────────────────────────────────────────────
def simple_router(query: str) -> tuple[Optional[str], Optional[str]]:
    """
    Keyword-based роутер.
    Возвращает (category, chunk_type) по тексту запроса.
    Переиспользован напрямую из test_retrieval.py.
    """
    q = query.lower()
    category   = None
    chunk_type = None

    if any(x in q for x in ["товар", "магазин", "продав", "чек", "дефект", "брак"]):
        category = "return_goods"
    elif any(x in q for x in ["жкх", "ук", "тсж", "отоплен", "квартплат", "вода", "гжи", "дом"]):
        category = "housing_utilities"

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


# ─────────────────────────────────────────────
# Retriever — адаптирован из test_retrieval.py
# ─────────────────────────────────────────────
class _Retriever:
    """
    Семантический ретривер на FAISS.
    Архитектура и фильтрация повторяют test_retrieval.py.
    """

    def __init__(self, index, meta: list[dict], model):
        self.index  = index
        self.meta   = meta
        self.model  = model

    def _embed(self, query: str) -> np.ndarray:
        return self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

    def _candidate_ids(
        self,
        category: Optional[str],
        chunk_type: Optional[str],
    ) -> list[int]:
        ids = []
        for i, chunk in enumerate(self.meta):
            ok_cat  = category   is None or chunk["category"]   == category   or chunk["category"] == "meta"
            ok_type = chunk_type is None or chunk["chunk_type"] == chunk_type or chunk["category"] == "meta"
            if ok_cat and ok_type:
                ids.append(i)
        return ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> list[dict]:
        """Точное повторение алгоритма из test_retrieval.py."""
        import faiss

        candidate_ids = self._candidate_ids(category=category, chunk_type=chunk_type)
        if not candidate_ids:
            return []

        dim = self.index.d
        candidate_matrix = np.empty((len(candidate_ids), dim), dtype="float32")
        for row_idx, chunk_idx in enumerate(candidate_ids):
            candidate_matrix[row_idx] = self.index.reconstruct(chunk_idx)

        sub_index = faiss.IndexFlatIP(dim)
        sub_index.add(candidate_matrix)

        q_vec = self._embed(query)
        scores, local_ids = sub_index.search(q_vec, min(top_k, len(candidate_ids)))

        results = []
        for score, local_id in zip(scores[0], local_ids[0]):
            global_id = candidate_ids[local_id]
            chunk = self.meta[global_id]
            results.append({
                "score":      float(score),
                "id":         chunk["id"],
                "section":    chunk["section"],
                "category":   chunk["category"],
                "chunk_type": chunk["chunk_type"],
                "title":      chunk["title"],
                "text":       chunk["text"],
            })
        return results


# ─────────────────────────────────────────────
# RAGEngine — публичный класс для app.py
# ─────────────────────────────────────────────
class RAGEngine:
    """
    Загружает FAISS-индекс (data/kb.faiss) и выполняет поиск.

    Если индекс не найден — падает в TF-IDF режим (scikit-learn).
    Для полноценной работы сначала запусти:
        python build_chunks.py
        python build_index.py
    """

    def __init__(self):
        # Пробуем загрузить мета из data/kb_meta.json (после build_index.py)
        # Иначе — из data/kb_chunks.json (после build_chunks.py)
        meta_source = META_PATH if META_PATH.exists() else CHUNKS_PATH
        if not meta_source.exists():
            raise FileNotFoundError(
                f"Не найден ни {META_PATH}, ни {CHUNKS_PATH}. "
                "Запустите build_chunks.py и build_index.py."
            )

        with open(meta_source, encoding="utf-8") as f:
            self.meta: list[dict] = json.load(f)

        self.chunks_with_text = [c for c in self.meta if c.get("text", "").strip()]
        self._retriever: Optional[_Retriever] = None
        self._backend = self._init_backend()

    # ── Инициализация ─────────────────────────
    def _init_backend(self) -> str:
        if INDEX_PATH.exists():
            try:
                import faiss
                from sentence_transformers import SentenceTransformer

                index = faiss.read_index(str(INDEX_PATH))
                model = SentenceTransformer(EMBED_MODEL)
                self._retriever = _Retriever(index=index, meta=self.meta, model=model)
                print(f"[RAG] Бэкенд: FAISS + SentenceTransformer ({len(self.meta)} чанков)")
                return "faiss"

            except Exception as e:
                print(f"[RAG] FAISS загрузить не удалось ({e.__class__.__name__}), используем TF-IDF")

        # TF-IDF fallback
        self._tfidf_build()
        print(f"[RAG] Бэкенд: TF-IDF ({len(self.chunks_with_text)} чанков)")
        return "tfidf"

    def _tfidf_build(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        texts = [c["text"] for c in self.chunks_with_text]
        self._tfidf_vec    = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self._tfidf_matrix = self._tfidf_vec.fit_transform(texts)
        self._cos_sim      = cosine_similarity

    # ── Поиск ─────────────────────────────────
    def retrieve(
        self,
        query: str,
        category_key: str,
        doc_type_key: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Двухэтапный поиск:
        1. Структурный — жёстко берём нужные chunk_type для категории/типа.
        2. Семантический — дополняем через FAISS/TF-IDF (используя simple_router).
        """
        priority_types = set(
            PRIORITY_CHUNKS.get(category_key, {}).get(doc_type_key, [])
        )

        # ── Этап 1: структурные чанки ─────────
        must_have: list[dict] = []
        seen_ids: set[str]    = set()

        for chunk in self.chunks_with_text:
            ctype = chunk["chunk_type"]
            cat   = chunk["category"]
            if cat == category_key and ctype in priority_types:
                must_have.append(chunk)
                seen_ids.add(chunk["id"])
            elif cat == "meta" and ctype in META_TYPES:
                must_have.append(chunk)
                seen_ids.add(chunk["id"])

        # ── Этап 2: семантическое дополнение ─
        semantic: list[dict] = []
        if query.strip():
            router_cat, router_type = simple_router(query)

            if self._backend == "faiss" and self._retriever:
                results = self._retriever.search(
                    query=query,
                    top_k=top_k + len(seen_ids),
                    category=router_cat,
                    chunk_type=router_type,
                )
                for r in results:
                    if r["id"] not in seen_ids and r["text"].strip():
                        semantic.append(r)
                        seen_ids.add(r["id"])
                        if len(semantic) >= top_k:
                            break

            else:  # tfidf
                q_vec  = self._tfidf_vec.transform([query])
                sims   = self._cos_sim(q_vec, self._tfidf_matrix)[0]
                ranked = np.argsort(sims)[::-1]
                for idx in ranked:
                    chunk = self.chunks_with_text[idx]
                    if chunk["id"] not in seen_ids:
                        # Применяем ту же категориальную фильтрацию, что в _Retriever
                        ok_cat  = router_cat  is None or chunk["category"]   == router_cat  or chunk["category"] == "meta"
                        ok_type = router_type is None or chunk["chunk_type"] == router_type or chunk["category"] == "meta"
                        if ok_cat and ok_type:
                            semantic.append(chunk)
                            seen_ids.add(chunk["id"])
                        if len(semantic) >= top_k:
                            break

        return must_have + semantic


# ─────────────────────────────────────────────
# build_rag_prompt — вызывается из app.py
# ─────────────────────────────────────────────
def _format_evidence(
    evidence_checked: list[str],
    evidence_details: dict,
) -> str:
    """
    Формирует блок доказательств для промпта.
    Если для доказательства заполнены детали — добавляет их.
    """
    if not evidence_checked:
        return "  - не указаны"

    lines = []
    for ev in evidence_checked:
        details = evidence_details.get(ev, {})
        if details:
            detail_parts = "; ".join(f"{k}: {v}" for k, v in details.items() if v)
            lines.append(f"  ✓ {ev}\n      Детали: {detail_parts}")
        else:
            lines.append(f"  ✓ {ev}")
    return "\n".join(lines)


def build_rag_prompt(
    category: str,
    doc_type: str,
    user_data: dict,
    respondent_data: dict,
    problem_text: str,
    evidence_checked: list[str],
    extra_data: dict,
    rag_engine: RAGEngine,
    evidence_details: dict | None = None,
) -> str:
    """Собирает финальный промпт: KB-контекст + данные пользователя + задача."""

    if evidence_details is None:
        evidence_details = {}

    category_key = CATEGORY_MAP.get(category, "return_goods")
    doc_type_key  = DOC_TYPE_MAP.get(doc_type, "pretrial")

    # Формируем поисковый запрос: ключевые слова + описание ситуации
    query = f"{category} {doc_type} {problem_text[:300]}"
    chunks = rag_engine.retrieve(
        query=query,
        category_key=category_key,
        doc_type_key=doc_type_key,
        top_k=4,
    )

    # KB-контекст
    kb_context = "\n\n".join(
        f"[{c['title']}]\n{c['text']}"
        for c in chunks if c.get("text", "").strip()
    )

    # Доказательства с деталями
    evidence_str = _format_evidence(evidence_checked, evidence_details)

    # Блок данных пользователя
    if category_key == "return_goods":
        user_block = f"""Данные заявителя:
  ФИО: {user_data['fio']}
  Адрес: {user_data['address']}
  Телефон: {user_data.get('phone', '[не указан]')}

Данные ответчика (продавца):
  Наименование: {respondent_data['name']}
  Адрес: {respondent_data['address']}

Данные о товаре:
  Наименование товара: {extra_data.get('item_name', '[не указано]')}
  Стоимость товара: {extra_data.get('item_price', '[не указано]')} руб.
  Дата покупки: {extra_data.get('purchase_date', '[не указана]')}
  Дата обнаружения недостатка: {extra_data.get('defect_date', '[не указана]')}
  Требование заявителя: {extra_data.get('demand', '[не указано]')}"""
    else:
        user_block = f"""Данные заявителя:
  ФИО: {user_data['fio']}
  Адрес: {user_data['address']}, кв. № {extra_data.get('apartment', '[?]')}
  Телефон: {user_data.get('phone', '[не указан]')}

Данные ответчика (УК/ТСЖ):
  Наименование: {respondent_data['name']}
  Адрес: {respondent_data['address']}

Данные о нарушении:
  Вид нарушения: {extra_data.get('violation_type', '[не указан]')}
  Дата нарушения: {extra_data.get('violation_date', '[не указана]')}"""

    # Задача
    if doc_type_key == "pretrial":
        task = (
            "Сгенерируй досудебную претензию на основе шаблона из базы знаний выше. "
            "Подставь данные пользователя вместо полей в фигурных скобках. "
            "Структура: шапка (кому/от кого), заголовок ПРЕТЕНЗИЯ, описание ситуации, "
            "требование со ссылкой на закон, последствия неисполнения, "
            "список приложений, дата и подпись."
        )
    else:
        task = (
            "Сгенерируй исковое заявление на основе шаблона из базы знаний выше. "
            "Подставь данные пользователя вместо полей в фигурных скобках. "
            "Структура: шапка (суд/истец/ответчик), заголовок ИСКОВОЕ ЗАЯВЛЕНИЕ, "
            "обстоятельства дела, правовое обоснование, просительная часть ПРОШУ, "
            "список приложений, дата и подпись."
        )

    return f"""Ты — юридический ассистент, помогающий гражданам России самостоятельно защищать свои права.

════════════════════════════════════════
БАЗА ЗНАНИЙ (шаблоны и правила)
════════════════════════════════════════
{kb_context}

════════════════════════════════════════
ДАННЫЕ ПОЛЬЗОВАТЕЛЯ
════════════════════════════════════════
Категория спора: {category}
Тип документа: {doc_type}

{user_block}

Описание ситуации (слова пользователя):
{problem_text}

Имеющиеся доказательства:
{evidence_str}

════════════════════════════════════════
ЗАДАЧА
════════════════════════════════════════
{task}

Правила генерации:
- Заполняй все поля {{…}} данными пользователя.
- Если данных нет — оставляй [ЗАПОЛНИТЬ], не придумывай факты.
- Не изменяй ссылки на статьи законов из шаблона.
- Описание недостатка/нарушения — формулировку пользователя переведи в официально-деловой стиль.
- Никаких разговорных выражений, сокращений и эмодзи.
- Выдай ТОЛЬКО текст документа, без пояснений до или после него."""
