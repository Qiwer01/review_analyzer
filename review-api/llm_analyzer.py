"""
llm_analyzer.py

AI-часть: LLM читает негативные отзывы (уже отфильтрованные ML-моделью),
группирует их по темам жалоб и предлагает бизнесу конкретные улучшения.

Ключ OPENAI_API_KEY ожидается в переменных окружения — в Colab он попадает
туда из Secrets (см. ноутбук), при Docker-запуске — через docker run -e.
"""

import os
import json
from typing import List, Dict

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DEFAULT_MODEL = "gpt-4o-mini"


PROMPT_TEMPLATE = """Ты — аналитик клиентского опыта для бизнеса, который продаёт продукты питания.
Ниже дана выборка НЕГАТИВНЫХ отзывов покупателей (каждый отзыв пронумерован).

Твоя задача:
1. Сгруппировать отзывы по 3-6 ключевым темам жалоб (например: качество продукта,
   доставка, упаковка, цена, вкус, испорченный товар и т.д.)
2. Для каждой темы:
   - короткое название
   - примерная доля отзывов по теме (%)
   - 1-2 предложения с сутью проблемы
   - один короткий пример (перефразированный, НЕ дословная цитата)
   - одна конкретная, выполнимая рекомендация для бизнеса
3. В конце — общий вывод: топ-3 приоритета для бизнеса.

Ответь СТРОГО в формате JSON, без markdown-разметки и пояснений до/после, по схеме:

{{
  "topics": [
    {{
      "title": "string",
      "share_percent": number,
      "description": "string",
      "example": "string",
      "recommendation": "string"
    }}
  ],
  "top_priorities": ["string", "string", "string"]
}}

Отзывы:
{reviews}
"""


def _format_reviews(reviews: List[str], max_reviews: int = 40) -> str:
    sample = reviews[:max_reviews]
    return "\n".join(f"{i + 1}. {r.strip()}" for i, r in enumerate(sample))


def analyze_negative_reviews(reviews: List[str], model: str = DEFAULT_MODEL) -> Dict:
    if not reviews:
        return {"topics": [], "top_priorities": []}

    prompt = PROMPT_TEMPLATE.format(reviews=_format_reviews(reviews))

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Не удалось распарсить ответ LLM как JSON", "raw_response": raw}


def format_report_markdown(analysis: Dict) -> str:
    if "error" in analysis:
        return (
            f"⚠️ Ошибка анализа: {analysis['error']}\n\n"
            f"```\n{analysis.get('raw_response', '')}\n```"
        )

    lines = ["# 📊 Отчёт по анализу негативных отзывов\n"]

    for topic in analysis.get("topics", []):
        lines.append(f"## {topic.get('title', 'Без названия')} "
                      f"({topic.get('share_percent', '?')}%)")
        lines.append(f"**Проблема:** {topic.get('description', '')}")
        lines.append(f"**Пример:** _{topic.get('example', '')}_")
        lines.append(f"**Рекомендация:** {topic.get('recommendation', '')}\n")

    if analysis.get("top_priorities"):
        lines.append("## 🎯 Топ-3 приоритета для бизнеса")
        for i, p in enumerate(analysis["top_priorities"], 1):
            lines.append(f"{i}. {p}")

    return "\n".join(lines)
