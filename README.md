# YouTube Multi-Label Toxicity Web App

React + TypeScript + Vite frontend with a FastAPI backend that loads the scikit-learn model exported from Google Colab.

## 1. Export the trained model from Colab

Your notebook should save this bundle:

```python
model_bundle = {
    "model": model,
    "labels": LABELS,
    "thresholds": thresholds,
}
joblib.dump(model_bundle, "youtube_multilabel_toxicity.joblib")
```

Copy `youtube_multilabel_toxicity.joblib` into:

```text
backend/youtube_multilabel_toxicity.joblib
```

## 2. Start the backend

Windows PowerShell:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

API: http://127.0.0.1:8000
Docs: http://127.0.0.1:8000/docs

## 3. Start the frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:5173

## API request

`POST /predict`

```json
{
  "comment": "You are an idiot and this video is complete garbage."
}
```

The response contains probability, threshold and detection status for all six labels.
