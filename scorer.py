"""
scorer.py — ARLC evaluation metrics

Scoring formula: Total = (0.7 * S_det + 0.3 * S_asst) * G * T * F

  S_det  — score for determinate answers (boolean, date, number, name, names)
  S_asst — score for free_text (LLM judge, 0 or 1)
  G      — grounding score: F-beta(precision, recall) on pages, β=2.5
  T      — timeliness: fraction of answers with total_time_ms ≤ 100000
  F      — completeness: fraction of non-null answers
"""

import re
from typing import Any


FBETA = 2.5  # β for F-beta


def fbeta(precision: float, recall: float, beta: float = FBETA) -> float:
    """F-beta score. β=2.5 weights recall more than precision."""
    if precision + recall == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * precision * recall / (b2 * precision + recall)


# ── Determinate answer scoring ────────────────────────────────────────────────

def _normalize_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip().lower()


def _norm_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        vl = v.strip().lower()
        if vl in ("true", "yes", "1"):
            return True
        if vl in ("false", "no", "0"):
            return False
    return None


def _norm_date(v: Any) -> str:
    """Normalize date to YYYY-MM-DD or empty string."""
    if v is None:
        return ""
    s = str(v).strip()
    # Already YYYY-MM-DD
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    return _normalize_str(s)


def _norm_number(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip().replace(",", ""))
    except Exception:
        return None


def _norm_names_set(v: Any) -> set[str]:
    """Normalize names list to a set of lowercased strings."""
    if v is None:
        return set()
    if isinstance(v, list):
        return {str(x).strip().lower() for x in v if x is not None}
    if isinstance(v, str):
        # Could be a single name
        return {v.strip().lower()}
    return set()


def score_det_one(answer_type: str, gold: Any, pred: Any) -> float:
    """Score a single determinate answer. Returns 0.0 or 1.0.

    If gold is None → correct answer is 'unanswerable', only None pred gets 1.0.
    """
    # Unanswerable question
    if gold is None:
        return 1.0 if pred is None else 0.0

    if pred is None:
        return 0.0

    if answer_type == "boolean":
        g = _norm_bool(gold)
        p = _norm_bool(pred)
        return 1.0 if (g is not None and p is not None and g == p) else 0.0

    elif answer_type == "date":
        g = _norm_date(gold)
        p = _norm_date(pred)
        return 1.0 if (g and p and g == p) else 0.0

    elif answer_type == "number":
        g = _norm_number(gold)
        p = _norm_number(pred)
        if g is None or p is None:
            return 0.0
        # Allow ±1% tolerance for floating point rounding
        if g == 0:
            return 1.0 if p == 0 else 0.0
        return 1.0 if abs(g - p) / abs(g) < 0.01 else 0.0

    elif answer_type == "name":
        g = _normalize_str(gold)
        p = _normalize_str(pred)
        return 1.0 if (g and p and g == p) else 0.0

    elif answer_type == "names":
        g_set = _norm_names_set(gold)
        p_set = _norm_names_set(pred)
        if not g_set:
            return 0.0
        tp = len(g_set & p_set)
        precision = tp / len(p_set) if p_set else 0.0
        recall = tp / len(g_set)
        return fbeta(precision, recall)

    return 0.0


# ── Grounding score ───────────────────────────────────────────────────────────

def _page_set(refs: list[dict]) -> set[tuple[str, int]]:
    """Convert refs list to set of (doc_id, page_num) tuples."""
    pages = set()
    for ref in (refs or []):
        doc_id = ref.get("doc_id", "")
        for pg in ref.get("pages", ref.get("page_numbers", [])):
            pages.add((doc_id, int(pg)))
    return pages


def score_grounding(gold_refs: list[dict], pred_refs: list[dict]) -> float:
    """Grounding score: F-beta between predicted and gold page sets."""
    gold_pages = _page_set(gold_refs)
    pred_pages = _page_set(pred_refs)

    if not gold_pages:
        # No gold refs — grounding can't be evaluated, give full credit
        return 1.0

    if not pred_pages:
        return 0.0

    tp = len(gold_pages & pred_pages)
    precision = tp / len(pred_pages)
    recall = tp / len(gold_pages)
    return fbeta(precision, recall)


# ── Main scorer ───────────────────────────────────────────────────────────────

def compute_scores(
    gold_map: dict,       # qid -> gold record (with answer, answer_type, refs)
    pred_map: dict,       # qid -> pred record (with answer, refs, timing_ms)
    asst_scores: dict,    # qid -> float (0.0 or 1.0) for free_text
    time_limit_ms: int = 100_000,
) -> dict:
    """
    Compute full ARLC score breakdown.

    Returns:
      {
        "total": float,
        "S_det": float, "S_asst": float,
        "G": float, "T": float, "F": float,
        "n_det": int, "n_asst": int,
        "details": [per-question results],
      }
    """
    det_scores = []
    asst_scores_list = []
    ground_scores = []
    timing_ok = []
    ttft_factors = []
    ttft_ms_list = []
    answered = 0
    total = len(gold_map)

    details = []

    for qid, gold_rec in gold_map.items():
        answer_type = gold_rec.get("answer_type", "free_text")
        gold_answer = gold_rec.get("answer")
        gold_refs = gold_rec.get("refs", [])

        pred_rec = pred_map.get(qid, {})
        pred_answer = pred_rec.get("answer")
        pred_refs = pred_rec.get("refs", [])
        timing = pred_rec.get("telemetry", {}).get("timing", {})
        total_time_ms = pred_rec.get("timing_ms") or timing.get("total_time_ms") or 0
        ttft_ms = timing.get("ttft_ms") or total_time_ms

        # Completeness
        if pred_answer is not None:
            answered += 1

        # Timeliness (T): telemetry validity
        t_ok = 1.0 if (total_time_ms is not None and total_time_ms <= time_limit_ms) else 0.0
        timing_ok.append(t_ok)

        # TTFT factor (F)
        if ttft_ms < 1000:
            f_factor = 1.05
        elif ttft_ms < 2000:
            f_factor = 1.02
        elif ttft_ms < 3000:
            f_factor = 1.00
        else:
            # Calibrated from warm-up: F=0.955 at mean_ttft=3928ms
            # Linear: 0.99 at 3000ms → 0.85 at 6713ms, floor 0.85
            f_factor = max(0.85, 0.99 - (ttft_ms - 3000) * 0.0000377)
        ttft_factors.append(f_factor)
        ttft_ms_list.append(ttft_ms)

        # Grounding
        g = score_grounding(gold_refs, pred_refs)
        ground_scores.append(g)

        # Answer score
        if answer_type == "free_text":
            s = asst_scores.get(qid, 0.0)
            asst_scores_list.append(s)
            det_s = None
        else:
            s = score_det_one(answer_type, gold_answer, pred_answer)
            det_scores.append(s)
            det_s = s

        details.append({
            "question_id": qid,
            "answer_type": answer_type,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "score": s,
            "det_score": det_s,
            "grounding": g,
            "timing_ok": t_ok,
        })

    S_det = (sum(det_scores) / len(det_scores)) if det_scores else 0.0
    S_asst = (sum(asst_scores_list) / len(asst_scores_list)) if asst_scores_list else 0.0
    G = (sum(ground_scores) / len(ground_scores)) if ground_scores else 0.0
    T = (sum(timing_ok) / len(timing_ok)) if timing_ok else 0.0
    F = (sum(ttft_factors) / len(ttft_factors)) if ttft_factors else 1.0
    mean_ttft_ms = round(sum(ttft_ms_list) / len(ttft_ms_list)) if ttft_ms_list else 0

    total_score = (0.7 * S_det + 0.3 * S_asst) * G * T * F

    return {
        "total": round(total_score, 4),
        "S_det": round(S_det, 4),
        "S_asst": round(S_asst, 4),
        "G": round(G, 4),
        "T": round(T, 4),
        "F": round(F, 4),
        "mean_ttft_ms": mean_ttft_ms,
        "n_det": len(det_scores),
        "n_asst": len(asst_scores_list),
        "n_answered": answered,
        "n_total": total,
        "details": details,
    }
