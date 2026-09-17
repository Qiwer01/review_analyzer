import re
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from llm_analyzer import analyze_negative_reviews, format_report_markdown

app = FastAPI(title="Аналитик отзывов для бизнеса", version="1.0")

# Модель грузится ОДИН раз при старте сервера, а не на каждый запрос
model = joblib.load("model.pkl")


def clean_text(text: str) -> str:
    """Та же функция, что использовалась при обучении модели в ноутбуке —
    если они разойдутся, модель увидит на входе не то, на чём училась."""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


class Review(BaseModel):
    text: str = Field(..., min_length=1, description="Текст отзыва")


class ReviewBatch(BaseModel):
    reviews: List[str] = Field(..., min_length=1, description="Список текстов отзывов")


@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/info")
def info():
    return {
        "model": "TF-IDF + LinearSVC",
        "classes": ["negative", "neutral", "positive"],
        "version": "1.0",
    }


@app.post("/predict")
def predict(review: Review):
    sentiment = model.predict([clean_text(review.text)])[0]
    return {"text": review.text, "sentiment": sentiment}


@app.post("/predict-batch")
def predict_batch(batch: ReviewBatch):
    cleaned = [clean_text(t) for t in batch.reviews]
    predictions = model.predict(cleaned)
    return {
        "results": [
            {"text": t, "sentiment": s} for t, s in zip(batch.reviews, predictions)
        ]
    }


@app.post("/analyze-negative")
def analyze_negative(batch: ReviewBatch):
    """Сам отбирает негативные отзывы через ML-модель и отправляет их в LLM."""
    cleaned = [clean_text(t) for t in batch.reviews]
    predictions = model.predict(cleaned)
    negative_reviews = [
        t for t, s in zip(batch.reviews, predictions) if s == "negative"
    ]

    if not negative_reviews:
        raise HTTPException(
            status_code=400,
            detail="Среди присланных отзывов не найдено негативных — анализировать нечего.",
        )

    analysis = analyze_negative_reviews(negative_reviews)
    report_md = format_report_markdown(analysis)
    return {
        "negative_count": len(negative_reviews),
        "analysis": analysis,
        "report_markdown": report_md,
    }
