from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "youtube_multilabel_toxicity.joblib"

app = FastAPI(
    title="YouTube Multi-Label Toxicity API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DISPLAY_NAMES = {
    "IsToxic": "Toxic",
    "IsAbusive": "Abusive",
    "IsProvocative": "Provocative",
    "IsHatespeech": "Hate Speech",
    "IsRacist": "Racist",
    "IsObscene": "Obscene",
}


class PredictionRequest(BaseModel):
    comment: str = Field(min_length=1, max_length=5000)


class LabelPrediction(BaseModel):
    key: str
    label: str
    probability: float
    threshold: float
    detected: bool


class PredictionResponse(BaseModel):
    comment: str
    predictions: list[LabelPrediction]


def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    bundle = load_bundle()
    return {
        "status": "ok" if bundle is not None else "model_missing",
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    comment = payload.comment.strip()
    if not comment:
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    bundle = load_bundle()
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model file is missing. Copy youtube_multilabel_toxicity.joblib "
                "into the backend folder."
            ),
        )

    model = bundle["model"]
    labels = bundle["labels"]
    thresholds = bundle["thresholds"]

    probabilities = model.predict_proba([comment])[0]

    predictions = []
    for key, probability, threshold in zip(labels, probabilities, thresholds):
        probability = float(probability)
        threshold = float(threshold)
        predictions.append(
            {
                "key": key,
                "label": DISPLAY_NAMES.get(key, key),
                "probability": probability,
                "threshold": threshold,
                "detected": probability >= threshold,
            }
        )

    return {
        "comment": comment,
        "predictions": predictions,
    }
