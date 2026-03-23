"""
judge.py — LLM-as-judge for free_text answers (ARLC edition)

Uses Grok to score free_text answers 0 or 1 against gold.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

XAI_API_KEY = os.getenv("XAI_API_KEY", "")
EVAL_MODEL = os.getenv("EVAL_MODEL", "grok-4-1-fast-non-reasoning")
CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "8"))

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
    return _client


SYSTEM_PROMPT = """\
You are a strict grader for the ARLC (Agentic RAG Legal Challenge) competition.
Score a free-text answer on 5 criteria, each 0 or 1.

Criteria:
1. Correctness   — all key facts from the gold answer are present, no factual errors
2. Completeness  — all aspects of the question are addressed, nothing important left out
3. Grounding     — no invented facts; every statement is plausible given the question domain
4. Calibration   — expresses appropriate uncertainty; does not overclaim or underclaim
5. Clarity       — clear, concise, directly answers the question; no filler or forbidden abbreviations
                   (Art./Regs./FIs/yrs/vs are forbidden; ≥/≤ symbols are forbidden)

Special cases:
- If gold says "There is no information on this question in the provided documents." and pred also says the same → all 5 criteria = 1
- If pred is empty or null → all 5 criteria = 0

Respond ONLY with valid JSON:
{"c1": 0|1, "c2": 0|1, "c3": 0|1, "c4": 0|1, "c5": 0|1, "reason": "one sentence"}
"""

USER_TEMPLATE = """\
Question: {question}

Gold answer: {gold}

Participant answer: {pred}
"""


def _parse_score(text: str) -> float:
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return 0.0
    try:
        obj = json.loads(m.group(0))
        # 5-criteria mode
        if "c1" in obj:
            total = sum(int(obj.get(f"c{i}", 0)) for i in range(1, 6))
            return round(total / 5, 2)
        # fallback: legacy score field
        return 1.0 if int(obj.get("score", 0)) == 1 else 0.0
    except Exception:
        return 0.0


def judge_one(qid: str, question: str, gold: str, pred: str) -> dict:
    pred = (pred or "").strip()
    if not pred:
        return {"question_id": qid, "score": 0, "reason": "empty answer"}

    prompt = USER_TEMPLATE.format(question=question, gold=gold, pred=pred)
    try:
        resp = _get_client().chat.completions.create(
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content or ""
        score = _parse_score(raw)
        reason = ""
        try:
            m2 = re.search(r'"reason"\s*:\s*"([^"]*)"', raw)
            reason = m2.group(1) if m2 else ""
        except Exception:
            pass
        return {"question_id": qid, "score": score, "reason": reason}
    except Exception as e:
        return {"question_id": qid, "score": 0, "reason": f"error: {e}"}


def judge_batch(items: list[dict], workers: int = CONCURRENCY) -> dict:
    """
    items: list of {question_id, question, gold, pred}
    Returns: {question_id -> score_float}
    """
    results = {}
    total = len(items)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                judge_one,
                x["question_id"],
                x.get("question", ""),
                str(x.get("gold", "")),
                str(x.get("pred", "") or ""),
            ): x["question_id"]
            for x in items
        }
        done = 0
        for future in as_completed(futures):
            qid = futures[future]
            try:
                res = future.result()
                results[qid] = float(res["score"])
            except Exception as e:
                results[qid] = 0.0
                print(f"[judge] error {qid}: {e}")
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  judge: {done}/{total}", flush=True)

    return results
