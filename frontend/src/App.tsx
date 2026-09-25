import { useMemo, useState } from 'react'
import './App.css'

type LabelPrediction = {
  key: string
  label: string
  probability: number
  threshold: number
  detected: boolean
}

type PredictionResponse = {
  comment: string
  predictions: LabelPrediction[]
}

const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000'

const examples = [
  'I really enjoyed this video, thank you for uploading it.',
  'You are an idiot and this video is complete garbage.',
]

function App() {
  const [comment, setComment] = useState('')
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const detectedCount = useMemo(
    () => result?.predictions.filter((item) => item.detected).length ?? 0,
    [result],
  )

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()

    const cleanComment = comment.trim()
    if (!cleanComment) {
      setError('Enter a comment first.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ comment: cleanComment }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail ?? 'Prediction failed.')
      }

      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="page-shell">
      <section className="hero">
        <div className="eyebrow">MULTI-LABEL NLP CLASSIFIER</div>
        <h1>YouTube Comment Toxicity Detector</h1>
        <p>
          Paste an English YouTube comment and analyze six toxicity categories
          using trained TF-IDF + One-vs-Rest Logistic Regression model.
        </p>
      </section>

      <section className="workspace">
        <form className="panel composer" onSubmit={handleSubmit}>
          <div className="panel-heading">
            <div>
              <span className="step">01</span>
              <h2>Analyze comment</h2>
            </div>
            <span className="counter">{comment.length}/5000</span>
          </div>

          <textarea
            value={comment}
            maxLength={5000}
            onChange={(event) => setComment(event.target.value)}
            placeholder="Type or paste a YouTube comment..."
            rows={9}
          />

          <div className="example-row">
            <span>Try:</span>
            {examples.map((example, index) => (
              <button
                className="example-button"
                key={example}
                type="button"
                onClick={() => {
                  setComment(example)
                  setResult(null)
                  setError('')
                }}
              >
                Example {index + 1}
              </button>
            ))}
          </div>

          {error && <div className="error-message">{error}</div>}

          <button className="analyze-button" type="submit" disabled={loading}>
            {loading ? 'Analyzing…' : 'Analyze toxicity'}
          </button>
        </form>

        <section className="panel results-panel">
          <div className="panel-heading">
            <div>
              <span className="step">02</span>
              <h2>Prediction</h2>
            </div>
            {result && (
              <span className={`summary-pill ${detectedCount > 0 ? 'warning' : ''}`}>
                {detectedCount} / 6 detected
              </span>
            )}
          </div>

          {!result ? (
            <div className="empty-state">
              <div className="empty-icon">↗</div>
              <h3>No analysis yet</h3>
              <p>Submit a comment to see probability scores for each label.</p>
            </div>
          ) : (
            <div className="results-list">
              {result.predictions.map((prediction) => (
                <article className="result-card" key={prediction.key}>
                  <div className="result-topline">
                    <div>
                      <strong>{prediction.label}</strong>
                      <span>
                        Threshold {(prediction.threshold * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="score-wrap">
                      <span className="score">
                        {(prediction.probability * 100).toFixed(1)}%
                      </span>
                      <span
                        className={`status ${prediction.detected ? 'detected' : 'clear'}`}
                      >
                        {prediction.detected ? 'YES' : 'NO'}
                      </span>
                    </div>
                  </div>

                  <div className="bar-track">
                    <div
                      className={`bar-fill ${prediction.detected ? 'detected' : ''}`}
                      style={{ width: `${Math.max(2, prediction.probability * 100)}%` }}
                    />
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      </section>

      <section className="model-note">
        <span>MODEL</span>
        <p>
          TF-IDF word + character features · One-vs-Rest Logistic Regression ·
          six independently thresholded labels
        </p>
      </section>
    </main>
  )
}

export default App
