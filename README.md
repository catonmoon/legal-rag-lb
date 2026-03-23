# ARLC Eval App

Local evaluator for the **Agentic RAG Legal Challenge (ARLC)** — final phase, 900 questions.

**Live demo:** https://manzherok.ru/eval/

---

## What it does

Upload a `submission.json` → get full scoring breakdown:

- **S_det** — accuracy on determinate types (boolean, date, number, name, names)
- **S_asst** — LLM judge score on free_text answers (5 criteria × 0/1)
- **G** — grounding: F-beta (β=2.5) between predicted refs and gold refs
- **T** — telemetry validity (fraction with total_time_ms ≤ 100 000)
- **Total** = (0.7 × S_det + 0.3 × S_asst) × G × T

Per-question breakdown table with correct/wrong/partial, predicted vs gold answers, grounding details.

---

## Stack

- **Gradio** — web UI
- **scorer.py** — deterministic scoring (S_det, G, T)
- **judge.py** — LLM judge for free_text via xAI Grok API
- **gold_answers_final.json** — 900 verified gold answers (manually curated)

---

## Run locally

```bash
pip install -r requirements.txt
export XAI_API_KEY="your-key"   # needed for LLM judge (free_text scoring)
python app.py
```

Opens at `http://localhost:7861`

---

## Docker

```bash
docker-compose up
```

---

## Submission format

```json
{
  "answers": [
    {
      "question_id": "...",
      "answer": "...",
      "telemetry": {
        "timing": {"ttft_ms": 1200, "total_time_ms": 3500},
        "refs": [{"doc_id": "...", "pages": [1, 2]}]
      }
    }
  ]
}
```
