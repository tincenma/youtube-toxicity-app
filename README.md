# YouTube Multi-Label Toxicity Detector

A full-stack machine learning application for detecting multiple forms of toxicity in English YouTube comments.

The project uses a **multi-label NLP classifier** trained on the YouTube Toxicity dataset. A single comment can be assigned to several toxicity categories at the same time, such as **toxic**, **abusive**, **hate speech**, or **racist**.

The repository includes:

* a **React + TypeScript + Vite** frontend;
* a **FastAPI** backend;
* a **scikit-learn** multi-label text classification model;
* per-label probability scores and optimized decision thresholds.

Repository:

https://github.com/tincenma/youtube-toxicity-app

## Features

* Analyze English YouTube comments in real time.
* Predict six toxicity labels simultaneously.
* Display a probability score for every label.
* Use a separate optimized decision threshold for each category.
* REST API powered by FastAPI.
* Interactive React interface.
* Automatic OpenAPI documentation through FastAPI.
* Lightweight classical NLP model that runs locally without a GPU.
* Ready for separate frontend/backend deployment.

## Toxicity Labels

The model predicts the following six labels:

| Label           | Description                             |
| --------------- | --------------------------------------- |
| `IsToxic`       | General toxic or harmful language       |
| `IsAbusive`     | Insulting, hostile, or abusive language |
| `IsProvocative` | Language intended to provoke or inflame |
| `IsHatespeech`  | Hate speech targeting a person or group |
| `IsRacist`      | Racist language or content              |
| `IsObscene`     | Obscene or strongly offensive language  |

This is a **multi-label classification** problem. One comment may receive multiple labels simultaneously.

Example:

```text
Comment: "You are an idiot and this video is complete garbage."

Toxic        71.7%   YES
Abusive      75.9%   YES
Provocative  21.8%   NO
Hate Speech  12.8%   NO
Racist       13.8%   NO
Obscene      24.9%   NO
```

## Machine Learning Pipeline

The current model is a classical NLP baseline built with scikit-learn.

```text
YouTube comment
      |
      v
Word-level TF-IDF
      +
Character-level TF-IDF
      |
      v
One-vs-Rest Logistic Regression
      |
      v
6 independent probability scores
      |
      v
Per-label optimized thresholds
      |
      v
Detected / Not Detected
```

### Text Representation

The classifier combines:

* **word-level TF-IDF** using unigrams and bigrams;
* **character-level TF-IDF** using character n-grams.

Character features are useful for informal online language, spelling variations, punctuation, and partially unseen words.

### Classification Strategy

A separate Logistic Regression classifier is trained for each toxicity label using the **One-vs-Rest** strategy.

Each classifier independently estimates the probability that its category is present.

Because the six labels have different class distributions, the model uses `class_weight="balanced"` during training.

### Decision Thresholds

Instead of applying a universal `0.50` threshold, the project optimizes a separate threshold for each label using the validation set.

The optimized thresholds are saved together with the model:

```python
model_bundle = {
    "model": model,
    "labels": LABELS,
    "thresholds": thresholds,
}

joblib.dump(
    model_bundle,
    "youtube_multilabel_toxicity.joblib"
)
```

## Model Performance

The following results were obtained on the **validation set** after tuning per-label thresholds.

> These are validation metrics, not final held-out test metrics.

### Overall Multi-Label Metrics

| Metric            |      Score |
| ----------------- | ---------: |
| Micro Precision   |     0.5104 |
| Micro Recall      |     0.7387 |
| Micro F1          | **0.6037** |
| Macro F1          | **0.5541** |
| Weighted F1       |     0.6064 |
| Hamming Loss      |     0.2159 |
| Jaccard Score     |     0.2974 |
| Exact Match Ratio |     0.4027 |

### Per-Label Performance

| Label       | Precision | Recall |        F1 |
| ----------- | --------: | -----: | --------: |
| Toxic       |     0.648 |  0.838 | **0.731** |
| Abusive     |     0.495 |  0.868 | **0.630** |
| Provocative |     0.382 |  0.565 | **0.456** |
| Hate Speech |     0.387 |  0.571 | **0.462** |
| Racist      |     0.357 |  0.526 | **0.426** |
| Obscene     |     0.643 |  0.600 | **0.621** |

The model performs best on the more frequent and broader categories such as `IsToxic` and `IsAbusive`. Performance is weaker on less frequent and more specific labels such as `IsRacist` and `IsHatespeech`.

## Dataset

This project uses the **YouTube Toxicity Data** dataset available on Kaggle:

https://www.kaggle.com/datasets/reihanenamdari/youtube-toxicity-data

The dataset contains approximately **1,000 manually labeled English YouTube comments** with multiple toxicity-related annotations.

The original dataset contains additional labels, but this project uses six labels with enough positive examples to support training and evaluation.

Labels with extremely few or zero positive examples were excluded from the current model.

Please review the dataset's Kaggle page for its original license, attribution requirements, and usage terms.

## Data Preparation

The preprocessing pipeline intentionally keeps most of the original comment text.

The cleaning process:

1. selects the comment text and the six target labels;
2. removes rows with missing text or labels;
3. trims leading and trailing whitespace;
4. removes empty comments;
5. converts boolean labels to `0` and `1`;
6. detects duplicate comments using normalized lowercase text;
7. removes duplicate groups before splitting the dataset.

The pipeline does **not** remove:

* punctuation;
* profanity;
* capitalization;
* numbers;
* slang;
* spelling variations.

These elements can contain useful information for toxicity detection.

The dataset is split into approximately:

* 70% training;
* 15% validation;
* 15% test.

Multi-label stratification is used to preserve label distributions across the splits.

## Tech Stack

### Frontend

* React
* TypeScript
* Vite

### Backend

* Python
* FastAPI
* Uvicorn
* Pydantic

### Machine Learning

* scikit-learn
* NumPy
* SciPy
* joblib
* TF-IDF
* One-vs-Rest Logistic Regression

## Project Structure

```text
youtube-toxicity-app/
|
|-- backend/
|   |-- main.py
|   |-- requirements.txt
|   |-- .gitignore
|   `-- youtube_multilabel_toxicity.joblib
|
|-- frontend/
|   |-- public/
|   |-- src/
|   |   |-- App.tsx
|   |   |-- App.css
|   |   |-- main.tsx
|   |   `-- index.css
|   |-- package.json
|   |-- package-lock.json
|   |-- vite.config.ts
|   `-- index.html
|
`-- README.md
```

## Getting Started

### Prerequisites

Install:

* Python 3.10 or newer
* Node.js 18 or newer
* npm

The trained model file is expected at:

```text
backend/youtube_multilabel_toxicity.joblib
```

## Backend Setup

Move into the backend directory:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the API:

```bash
python -m uvicorn main:app --reload
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
GET http://127.0.0.1:8000/health
```

## Frontend Setup

Open another terminal and move into the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the Vite development server:

```bash
npm run dev
```

Open:

```text
http://localhost:5173
```

## Frontend Environment Variable

By default, the frontend sends requests to:

```text
http://127.0.0.1:8000
```

To use another backend URL, create a `.env` file inside `frontend/`:

```env
VITE_API_URL=https://your-api.example.com
```

Restart the Vite development server after changing environment variables.

## API

### `POST /predict`

Analyzes one comment.

Request:

```json
{
  "comment": "You are an idiot and this video is complete garbage."
}
```

Example response:

```json
{
  "comment": "You are an idiot and this video is complete garbage.",
  "predictions": [
    {
      "key": "IsToxic",
      "label": "Toxic",
      "probability": 0.717,
      "threshold": 0.46,
      "detected": true
    },
    {
      "key": "IsAbusive",
      "label": "Abusive",
      "probability": 0.759,
      "threshold": 0.37,
      "detected": true
    }
  ]
}
```

Each prediction contains:

* `probability` — model-estimated probability for the label;
* `threshold` — validation-optimized decision threshold;
* `detected` — whether the probability exceeds the label threshold.

## Deployment

The recommended production setup is:

```text
User
  |
  v
Vercel
React + Vite frontend
  |
  | POST /predict
  v
Render
FastAPI backend
  |
  v
youtube_multilabel_toxicity.joblib
```

### 1. Prepare the Repository

Do not commit local Python environments or cache files.

A root `.gitignore` should include:

```gitignore
# Python
.venv/
venv/
__pycache__/
*.py[cod]

# Node
node_modules/
dist/

# Environment
.env
.env.*
!.env.example

# Editors / OS
.vscode/
.idea/
.DS_Store
```

If `.venv` or `__pycache__` were already committed, remove them from Git tracking:

```bash
git rm -r --cached backend/.venv
git rm -r --cached backend/__pycache__
```

Then commit the cleanup:

```bash
git add .
git commit -m "chore: prepare app for deployment"
git push
```

The trained model file can remain in the repository:

```text
backend/youtube_multilabel_toxicity.joblib
```

### 2. Configure Production CORS

The backend must allow requests from the deployed frontend domain.

A deployment-friendly CORS configuration can use an environment variable:

```python
import os

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

This allows local development by default while letting the production frontend URL be configured on the hosting platform.

### 3. Deploy the FastAPI Backend on Render

Create a new **Web Service** in Render and connect this GitHub repository.

Recommended settings:

| Setting           | Value                                          |
| ----------------- | ---------------------------------------------- |
| Branch            | `main`                                         |
| Root Directory    | `backend`                                      |
| Runtime           | Python                                         |
| Build Command     | `pip install -r requirements.txt`              |
| Start Command     | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/health`                                      |

After deployment, Render will provide a URL similar to:

```text
https://youtube-toxicity-api.onrender.com
```

Check the backend:

```text
https://youtube-toxicity-api.onrender.com/health
```

Expected response:

```json
{
  "status": "ok",
  "model_path": "..."
}
```

FastAPI documentation will be available at:

```text
https://youtube-toxicity-api.onrender.com/docs
```

### 4. Deploy the React Frontend on Vercel

Create a new Vercel project from the same GitHub repository.

Use:

| Setting          | Value           |
| ---------------- | --------------- |
| Root Directory   | `frontend`      |
| Framework Preset | Vite            |
| Install Command  | `npm install`   |
| Build Command    | `npm run build` |
| Output Directory | `dist`          |

Before deploying, add the environment variable:

```text
VITE_API_URL=https://youtube-toxicity-api.onrender.com
```

Use your actual Render backend URL.

Vercel will provide a frontend URL similar to:

```text
https://youtube-toxicity-app.vercel.app
```

### 5. Add the Frontend URL to Render

Once the Vercel domain is known, add the following environment variable to the Render service:

```text
ALLOWED_ORIGINS=https://youtube-toxicity-app.vercel.app
```

For both production and local development:

```text
ALLOWED_ORIGINS=https://youtube-toxicity-app.vercel.app,http://localhost:5173,http://127.0.0.1:5173
```

Redeploy the backend after changing the environment variable.

### 6. Verify the Deployment

Check the backend first:

```text
GET https://your-render-service.onrender.com/health
```

Then test the prediction endpoint:

```bash
curl -X POST \
  https://your-render-service.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"comment":"You are an idiot and this video is complete garbage."}'
```

Finally, open the Vercel frontend and submit a comment through the UI.

### Deployment Environment Variables

#### Frontend / Vercel

```env
VITE_API_URL=https://your-render-service.onrender.com
```

#### Backend / Render

```env
ALLOWED_ORIGINS=https://your-vercel-app.vercel.app
```

## Limitations

This project is intended primarily for **education, experimentation, and demonstration**.

Important limitations:

* The training dataset is small.
* The model was trained on English-language comments.
* Toxicity is contextual and subjective.
* Sarcasm, coded language, reclaimed language, and unusual spelling can be difficult to classify.
* Less frequent labels have lower predictive performance.
* A probability score should not be interpreted as certainty.
* Validation-tuned thresholds may not generalize equally well to other platforms or communities.

The model should **not be used as the sole basis for automatically banning, punishing, or restricting users**.

For real moderation systems, predictions should be combined with human review, policy context, larger representative datasets, and continuous evaluation.

## Possible Improvements

Future work could include:

* evaluating the final model on the untouched test split;
* fine-tuning DistilBERT or another Transformer model;
* comparing classical ML and Transformer performance;
* collecting a larger and more diverse dataset;
* supporting additional toxicity labels;
* multilingual toxicity detection;
* model explainability;
* batch comment analysis;
* confidence calibration;
* Docker support;
* CI/CD;
* production monitoring.

## Contributing

Contributions are welcome.

A typical workflow:

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feat/your-feature
```

3. Make your changes.
4. Commit your work.
5. Push the branch.
6. Open a pull request.

Bug fixes, UI improvements, model experiments, documentation updates, and evaluation improvements are welcome.

## Responsible Use

Toxicity detection systems can reproduce biases present in their training data.

Please use this project responsibly. Do not treat predictions as objective judgments about a person or community, and do not deploy the model for high-impact moderation decisions without appropriate human oversight and evaluation.

## License

If you intend to publish this repository as open source, add an explicit open-source license to the repository.

The **MIT License** is a common choice for educational software projects, but you should choose the license that matches how you want others to use and redistribute your code.

The dataset has its own licensing and usage terms and is not automatically covered by the software license of this repository.

## Acknowledgements

* YouTube Toxicity Data dataset on Kaggle
  https://www.kaggle.com/datasets/reihanenamdari/youtube-toxicity-data
* scikit-learn
* FastAPI
* React
* Vite
* Render
* Vercel

---

If you use this project for research or coursework, document the dataset version, preprocessing steps, model configuration, validation procedure, and final test results so the experiment can be reproduced.
