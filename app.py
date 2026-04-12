import streamlit as st
import requests
import json
from datetime import date

from rag_engine import RAGEngine, build_rag_prompt

@st.cache_resource
def get_rag_engine():
    return RAGEngine()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Цифровой правозащитник",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@400;700;900&family=Onest:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Onest', sans-serif;
}

.stApp {
    background: #0d1117;
    color: #e8edf3;
}

/* Header */
.hero-block {
    background: linear-gradient(135deg, #0d1117 0%, #0f2027 50%, #1a3a5c 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 36px 40px 28px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-block::before {
    content: '⚖';
    position: absolute;
    right: 32px;
    top: 16px;
    font-size: 96px;
    opacity: 0.07;
    line-height: 1;
}
.hero-title {
    font-family: 'Unbounded', sans-serif;
    font-size: 2rem;
    font-weight: 900;
    color: #4da6ff;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 0.95rem;
    color: #7a9bb5;
    margin: 0;
    font-weight: 300;
}

/* Section headers */
.section-label {
    font-family: 'Unbounded', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4da6ff;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e3a5f;
}

/* Cards */
.card {
    background: #111827;
    border: 1px solid #1e2d40;
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 16px;
}

/* Category buttons */
.cat-btn-active {
    background: linear-gradient(135deg, #1a4a7a, #0f3058) !important;
    border: 2px solid #4da6ff !important;
    border-radius: 12px !important;
}

/* Evidence checkboxes */
.evidence-block {
    background: #0d1f33;
    border-left: 3px solid #4da6ff;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 8px;
}
.evidence-block-optional {
    background: #1a1f2e;
    border-left: 3px solid #2d4a6b;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 8px;
}

/* Result area */
.result-card {
    background: #0d1f33;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 28px;
    font-family: 'Onest', sans-serif;
    font-size: 0.9rem;
    line-height: 1.8;
    white-space: pre-wrap;
    color: #d4e4f5;
}

/* Streamlit overrides */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div {
    background: #111827 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #e8edf3 !important;
    font-family: 'Onest', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #4da6ff !important;
    box-shadow: 0 0 0 2px rgba(77,166,255,0.15) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1a6bc4, #0d4a8f) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Unbounded', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 14px 28px !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2280e0, #1560b0) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(77,166,255,0.3) !important;
}

.stCheckbox > label {
    color: #b8d0e8 !important;
    font-size: 0.88rem !important;
    font-family: 'Onest', sans-serif !important;
}

div[data-testid="stRadio"] > label {
    color: #7a9bb5 !important;
    font-family: 'Unbounded', sans-serif !important;
    font-size: 0.65rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}

.stRadio > div {
    gap: 12px !important;
}

.stRadio > div > label {
    background: #111827 !important;
    border: 1px solid #1e2d40 !important;
    border-radius: 10px !important;
    padding: 14px 20px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    color: #e8edf3 !important;
    font-family: 'Onest', sans-serif !important;
    font-size: 0.9rem !important;
}
.stRadio > div > label:hover {
    border-color: #4da6ff !important;
    background: #0d1f33 !important;
}

.stDateInput > div > div > input {
    background: #111827 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #e8edf3 !important;
}

/* Tag pills */
.tag {
    display: inline-block;
    background: #0d2d4d;
    color: #4da6ff;
    border: 1px solid #1e4a7a;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-family: 'Unbounded', sans-serif;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-right: 8px;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e3a5f, transparent);
    margin: 20px 0;
}

/* Success message */
.success-pill {
    background: #0d3320;
    border: 1px solid #1a6640;
    color: #4caf7d;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.85rem;
    margin-bottom: 16px;
}

/* Warning */
.warn-pill {
    background: #2d1f00;
    border: 1px solid #6b4a00;
    color: #ffa726;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.85rem;
    margin-bottom: 16px;
}

/* Step badge */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: #1a4a7a;
    border: 1px solid #4da6ff;
    border-radius: 50%;
    font-family: 'Unbounded', sans-serif;
    font-size: 0.65rem;
    font-weight: 900;
    color: #4da6ff;
    margin-right: 10px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA: Evidence lists per category
# ─────────────────────────────────────────────
EVIDENCE = {
    "Возврат испорченного товара": {
        "required": [
            "Кассовый или товарный чек",
            "Фото/видео дефекта товара",
            "Сам товар (или его остатки)",
            "Гарантийный талон (при наличии)",
        ],
        "optional": [
            "Переписка с продавцом (скриншоты)",
            "Акт/заключение сервисного центра",
            "Свидетельские показания",
            "Скриншот карточки товара с сайта",
            "Выписка с банковской карты об оплате",
        ]
    },
    "Споры с ЖКХ": {
        "required": [
            "Договор управления с УК / Устав ТСЖ",
            "Квитанции об оплате за спорный период",
            "Акт о нарушении (с подписями соседей или УК)",
            "Фото/видео нарушения с датой и временем",
        ],
        "optional": [
            "Ответ УК на предыдущие обращения",
            "Результаты замеров (температура, давление и пр.)",
            "Ответ ГЖИ на жалобу",
            "Показания соседей",
            "Заключение эксперта (при ущербе имуществу)",
            "Распечатка норматива из ПП РФ № 354",
        ]
    }
}

# ─────────────────────────────────────────────
# DATA: Fields per category
# ─────────────────────────────────────────────
CATEGORY_HINTS = {
    "Возврат испорченного товара": {
        "respondent_label": "Продавец / магазин / ИП",
        "respondent_hint": "Например: ООО «Эльдорадо», ИП Иванов А.А.",
        "problem_hint": "Опишите: что купили, когда купили, какой обнаружен дефект, когда обнаружили.",
        "extra_fields": ["item_name", "item_price", "purchase_date", "defect_date"]
    },
    "Споры с ЖКХ": {
        "respondent_label": "Управляющая компания / ТСЖ",
        "respondent_hint": "Например: ООО «УК Жилсервис», ТСЖ «Наш дом»",
        "problem_hint": "Опишите: какая услуга нарушена, с какого момента, какой ущерб причинён, что уже делали.",
        "extra_fields": ["apartment", "violation_type", "violation_date"]
    }
}

VIOLATION_TYPES = [
    "Ненадлежащее отопление",
    "Нарушение качества горячей/холодной воды",
    "Неверное начисление платы",
    "Аварийное состояние общего имущества",
    "Игнорирование заявок жильцов",
    "Иное нарушение",
]


# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────
def build_prompt(category, doc_type, user_data, respondent_data,
                 problem_text, evidence_checked, extra_data):

    evidence_str = "\n".join([f"  - {e}" for e in evidence_checked]) if evidence_checked else "  - не указаны"

    if category == "Возврат испорченного товара":
        context_block = f"""
Категория спора: Возврат некачественного товара
Тип документа: {doc_type}

Данные заявителя:
  ФИО: {user_data['fio']}
  Адрес: {user_data['address']}
  Телефон: {user_data['phone']}

Данные ответчика (продавца):
  Наименование: {respondent_data['name']}
  Адрес: {respondent_data['address']}

Данные о товаре:
  Наименование товара: {extra_data.get('item_name', '[не указано]')}
  Стоимость товара: {extra_data.get('item_price', '[не указано]')} руб.
  Дата покупки: {extra_data.get('purchase_date', '[не указана]')}
  Дата обнаружения недостатка: {extra_data.get('defect_date', '[не указана]')}

Требование заявителя: {extra_data.get('demand', '[не указано]')}

Описание ситуации от пользователя:
{problem_text}

Имеющиеся доказательства:
{evidence_str}

Применимые нормы:
  - Закон РФ «О защите прав потребителей», ст. 18, 22, 23, 13
  - ГК РФ (договор купли-продажи)
  - ГПК РФ, ст. 131–132 (для иска)
"""
    else:
        context_block = f"""
Категория спора: Спор с ЖКХ / управляющей компанией
Тип документа: {doc_type}

Данные заявителя:
  ФИО: {user_data['fio']}
  Адрес: {user_data['address']} (квартира № {extra_data.get('apartment', '?')})
  Телефон: {user_data['phone']}

Данные ответчика (УК/ТСЖ):
  Наименование: {respondent_data['name']}
  Адрес: {respondent_data['address']}

Данные о нарушении:
  Вид нарушения: {extra_data.get('violation_type', '[не указан]')}
  Дата нарушения: {extra_data.get('violation_date', '[не указана]')}

Описание ситуации от пользователя:
{problem_text}

Имеющиеся доказательства:
{evidence_str}

Применимые нормы:
  - ЖК РФ, ст. 157, 161
  - Постановление Правительства РФ № 354 (правила предоставления КУ)
  - Постановление Правительства РФ № 491 (содержание общего имущества)
  - Закон РФ «О защите прав потребителей», ст. 29, 31, 13
  - ГПК РФ, ст. 131–132 (для иска)
"""

    if doc_type == "Досудебная претензия":
        task = "Сгенерируй досудебную претензию. Используй официально-деловой стиль. Укажи конкретные статьи законов из раздела 'Применимые нормы'. Если каких-то данных не хватает — оставь поле в виде [ЗАПОЛНИТЬ]. Не придумывай факты. Структура: шапка (кому/от кого), заголовок ПРЕТЕНЗИЯ, описание ситуации, требование, последствия неисполнения, список приложений, дата и подпись."
    else:
        task = "Сгенерируй исковое заявление. Используй официально-деловой стиль. Укажи конкретные статьи законов из раздела 'Применимые нормы'. Если каких-то данных не хватает — оставь поле в виде [ЗАПОЛНИТЬ]. Не придумывай факты. Структура: шапка (суд/истец/ответчик), заголовок ИСКОВОЕ ЗАЯВЛЕНИЕ с подзаголовком, обстоятельства дела, правовое обоснование, просительная часть ПРОШУ, список приложений, дата и подпись."

    return f"""Ты — юридический ассистент, помогающий гражданам России самостоятельно защищать свои права.

{context_block}

Задача: {task}

ВАЖНО: Не добавляй никаких пояснений до или после документа. Выдай только текст документа."""


# ─────────────────────────────────────────────
# API CALL (T-Pro / любая OpenAI-совместимая)
# ─────────────────────────────────────────────
def call_model(prompt: str, api_url: str, api_key: str, model_name: str) -> str:
    """
    Вызов модели через OpenAI-совместимый API.
    T-Pro 2.0 на Hugging Face или локальный vLLM — используй этот блок.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.3,
    }
    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        return "❌ Ошибка подключения к API. Проверьте URL модели."
    except requests.exceptions.Timeout:
        return "❌ Превышено время ожидания ответа от модели."
    except Exception as e:
        return f"❌ Ошибка: {str(e)}"


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "generated_doc" not in st.session_state:
    st.session_state.generated_doc = None
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = None


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-block">
    <div class="hero-title">Цифровой правозащитник</div>
    <p class="hero-sub">AI-сервис для самостоятельной защиты прав · Претензии и иски без юриста</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR: API Settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">⚙ Настройки API</div>', unsafe_allow_html=True)
    api_url = st.text_input(
        "URL модели",
        value="http://localhost:8000/v1/chat/completions",
        help="OpenAI-совместимый endpoint (vLLM, HF TGI, etc.)"
    )
    api_key = st.text_input("API ключ", value="token-abc123", type="password")
    model_name = st.text_input("Модель", value="t-pro-2.0")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">ℹ О проекте</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.8rem; color:#5a7a96; line-height:1.6;">
    Используется RAG + LLM для генерации юридических документов.<br><br>
    Документы носят <b style="color:#ffa726;">информационный характер</b> и не заменяют юридическую консультацию.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN LAYOUT: two columns
# ─────────────────────────────────────────────
col_form, col_result = st.columns([1, 1], gap="large")

with col_form:

    # ── STEP 1: Category ──────────────────────
    st.markdown('<div class="section-label">① Категория спора</div>', unsafe_allow_html=True)
    category = st.radio(
        "Выберите категорию",
        options=["Возврат испорченного товара", "Споры с ЖКХ"],
        label_visibility="collapsed"
    )
    hint = CATEGORY_HINTS[category]
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── STEP 2: Document type ────────────────
    st.markdown('<div class="section-label">② Тип документа</div>', unsafe_allow_html=True)
    doc_type = st.radio(
        "Тип документа",
        options=["Досудебная претензия", "Исковое заявление"],
        label_visibility="collapsed"
    )
    if doc_type == "Исковое заявление":
        st.markdown("""
        <div class="warn-pill">
        ⚠️ Иск подаётся после отказа удовлетворить претензию или истечения срока ответа.
        </div>
        """, unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── STEP 3: User data ────────────────────
    st.markdown('<div class="section-label">③ Ваши данные</div>', unsafe_allow_html=True)
    u_col1, u_col2 = st.columns(2)
    with u_col1:
        fio = st.text_input("ФИО *", placeholder="Иванова Мария Петровна")
        phone = st.text_input("Телефон", placeholder="+7 (999) 123-45-67")
    with u_col2:
        address = st.text_input("Адрес проживания *", placeholder="г. Новосибирск, ул. Ленина, д. 1, кв. 5")

    # ── STEP 4: Respondent ───────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-label">④ {hint["respondent_label"]}</div>', unsafe_allow_html=True)
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        resp_name = st.text_input("Наименование *", placeholder=hint["respondent_hint"])
    with r_col2:
        resp_address = st.text_input("Адрес ответчика *", placeholder="г. Новосибирск, ул. Мира, д. 10")

    # ── STEP 5: Category-specific fields ────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    extra_data = {}

    if category == "Возврат испорченного товара":
        st.markdown('<div class="section-label">⑤ Данные о товаре</div>', unsafe_allow_html=True)
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            extra_data["item_name"] = st.text_input("Наименование товара *", placeholder="Телевизор Samsung UE55")
            extra_data["purchase_date"] = str(st.date_input("Дата покупки", value=date.today()))
        with t_col2:
            extra_data["item_price"] = st.text_input("Стоимость (руб.) *", placeholder="35000")
            extra_data["defect_date"] = str(st.date_input("Дата обнаружения дефекта", value=date.today()))
        extra_data["demand"] = st.selectbox(
            "Требование *",
            ["Возврат денежных средств", "Замена товара", "Безвозмездное устранение дефекта", "Уменьшение цены"]
        )

    else:  # ЖКХ
        st.markdown('<div class="section-label">⑤ Данные о нарушении</div>', unsafe_allow_html=True)
        z_col1, z_col2 = st.columns(2)
        with z_col1:
            extra_data["apartment"] = st.text_input("Номер квартиры *", placeholder="42")
            extra_data["violation_type"] = st.selectbox("Вид нарушения *", VIOLATION_TYPES)
        with z_col2:
            extra_data["violation_date"] = str(st.date_input("Дата нарушения", value=date.today()))

    # ── STEP 6: Problem description ──────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">⑥ Описание ситуации</div>', unsafe_allow_html=True)
    problem_text = st.text_area(
        "Опишите вашу ситуацию своими словами *",
        placeholder=hint["problem_hint"],
        height=130
    )
    char_count = len(problem_text)
    st.markdown(
        f'<div style="font-size:0.75rem; color:{"#4caf7d" if char_count >= 50 else "#7a9bb5"}; margin-top:-8px;">'
        f'{"✓ Достаточно деталей" if char_count >= 50 else f"{max(0, 50 - char_count)} символов для минимального описания"}'
        f'</div>', unsafe_allow_html=True
    )

    # ── STEP 7: Evidence checkboxes ──────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">⑦ Наличие доказательств</div>', unsafe_allow_html=True)

    evidence_data = EVIDENCE[category]
    evidence_checked = []

    st.markdown("**Обязательные:**")
    for ev in evidence_data["required"]:
        col_check, col_label = st.columns([0.08, 0.92])
        with col_check:
            checked = st.checkbox("", key=f"req_{ev}", label_visibility="collapsed")
        with col_label:
            st.markdown(
                f'<div class="evidence-block">🔵 {ev}</div>',
                unsafe_allow_html=True
            )
        if checked:
            evidence_checked.append(ev)

    st.markdown("**Дополнительные (усиливают позицию):**")
    for ev in evidence_data["optional"]:
        col_check, col_label = st.columns([0.08, 0.92])
        with col_check:
            checked = st.checkbox("", key=f"opt_{ev}", label_visibility="collapsed")
        with col_label:
            st.markdown(
                f'<div class="evidence-block-optional">⚪ {ev}</div>',
                unsafe_allow_html=True
            )
        if checked:
            evidence_checked.append(ev)

    required_count = sum(
        1 for ev in evidence_data["required"]
        if st.session_state.get(f"req_{ev}", False)
    )
    total_required = len(evidence_data["required"])
    if required_count < total_required:
        st.markdown(
            f'<div class="warn-pill">⚠️ Отмечено {required_count} из {total_required} обязательных доказательств. '
            f'Это может ослабить позицию.</div>',
            unsafe_allow_html=True
        )

    # ── GENERATE BUTTON ──────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    required_filled = bool(fio.strip() and address.strip() and resp_name.strip() and problem_text.strip())

    if st.button("⚖ СФОРМИРОВАТЬ ДОКУМЕНТ", disabled=not required_filled):
        if not required_filled:
            st.error("Заполните все обязательные поля (*)")
        else:
            user_data = {"fio": fio, "address": address, "phone": phone}
            respondent_data = {"name": resp_name, "address": resp_address}

            rag_engine = get_rag_engine()

            prompt = build_rag_prompt(
                category=category,
                doc_type=doc_type,
                user_data=user_data,
                respondent_data=respondent_data,
                problem_text=problem_text,
                evidence_checked=evidence_checked,
                extra_data=extra_data,
                rag_engine=rag_engine,
            )

            st.session_state.last_prompt = prompt

            with st.spinner("Формируется документ..."):
                result = call_model(prompt, api_url, api_key, model_name)
                st.session_state.generated_doc = result

    if not required_filled:
        missing = []
        if not fio.strip(): missing.append("ФИО")
        if not address.strip(): missing.append("адрес")
        if not resp_name.strip(): missing.append("наименование ответчика")
        if not problem_text.strip(): missing.append("описание ситуации")
        if missing:
            st.markdown(
                f'<div style="font-size:0.78rem; color:#5a7a96; margin-top:6px;">Не заполнено: {", ".join(missing)}</div>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────
# RESULT COLUMN
# ─────────────────────────────────────────────
with col_result:
    st.markdown('<div class="section-label">Сгенерированный документ</div>', unsafe_allow_html=True)

    if st.session_state.generated_doc:
        doc_text = st.session_state.generated_doc

        st.markdown("""
        <div class="success-pill">
        ✓ Документ сформирован · Проверьте все поля и при необходимости уточните детали
        </div>
        """, unsafe_allow_html=True)

        # Metadata tags
        st.markdown(
            f'<span class="tag">{category[:10]}...</span>'
            f'<span class="tag">{doc_type}</span>'
            f'<span class="tag">{date.today().strftime("%d.%m.%Y")}</span>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Document text
        st.markdown(
            f'<div class="result-card">{doc_text}</div>',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Action buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            st.download_button(
                label="⬇ Скачать .txt",
                data=doc_text,
                file_name=f"document_{date.today().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
        with btn_col2:
            if st.button("🔄 Сбросить"):
                st.session_state.generated_doc = None
                st.session_state.last_prompt = None
                st.rerun()

        # Show prompt (debug)
        with st.expander("🔍 Промпт (для отладки)", expanded=False):
            st.code(st.session_state.last_prompt, language="text")

    else:
        # Empty state
        st.markdown("""
        <div style="
            border: 1px dashed #1e3a5f;
            border-radius: 12px;
            padding: 60px 32px;
            text-align: center;
            color: #2d4a6b;
            margin-top: 8px;
        ">
            <div style="font-size: 3rem; margin-bottom: 16px; opacity: 0.4;">⚖</div>
            <div style="font-family: 'Unbounded', sans-serif; font-size: 0.8rem; font-weight: 700; letter-spacing: 1px; margin-bottom: 8px;">
                ДОКУМЕНТ ПОЯВИТСЯ ЗДЕСЬ
            </div>
            <div style="font-size: 0.82rem; line-height: 1.6; color: #3a5a7a;">
                Заполните форму слева и нажмите<br>«Сформировать документ»
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Checklist hint
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Что нужно для хорошего документа</div>', unsafe_allow_html=True)
        steps = [
            ("①", "Выберите категорию и тип документа"),
            ("②", "Укажите ваши данные и данные ответчика"),
            ("③", "Заполните данные о товаре или нарушении"),
            ("④", "Подробно опишите ситуацию (чем больше деталей — тем точнее документ)"),
            ("⑤", "Отметьте имеющиеся доказательства"),
        ]
        for num, text in steps:
            st.markdown(f"""
            <div style="display:flex; align-items:flex-start; margin-bottom:12px; gap:10px;">
                <div style="
                    min-width:28px; height:28px;
                    background:#0d2d4d; border:1px solid #1e4a7a;
                    border-radius:50%; display:flex; align-items:center; justify-content:center;
                    font-family:'Unbounded',sans-serif; font-size:0.6rem; color:#4da6ff; font-weight:900;
                ">{num}</div>
                <div style="font-size:0.85rem; color:#5a7a96; padding-top:4px;">{text}</div>
            </div>
            """, unsafe_allow_html=True)
