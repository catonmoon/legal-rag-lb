#!/usr/bin/env python3
"""
eval.py — ARLC local evaluator

Usage:
  python eval.py --submission submission_final_v1.json [--gold gold_answers_final.json]
                 [--no-judge] [--out results.json]

Scoring formula: Total = (0.7 * S_det + 0.3 * S_asst) * G * T * F
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT_CHALLENGE = Path(__file__).resolve().parent.parent / "starter_kit"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorer import compute_scores
from judge import judge_batch


def load_submission(path: str) -> dict:
    """Load submission_final_v1.json → {qid: {answer, refs, timing_ms}}"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    answers = data if isinstance(data, list) else data.get("answers", [])
    result = {}
    for a in answers:
        qid = a.get("question_id", a.get("id", ""))
        tel = a.get("telemetry", {})
        timing = tel.get("timing", {}).get("total_time_ms", 0)
        # Extract refs from retrieval.retrieved_chunk_pages
        raw_pages = tel.get("retrieval", {}).get("retrieved_chunk_pages", [])
        refs = []
        for rp in raw_pages:
            refs.append({
                "doc_id": rp.get("doc_id", ""),
                "pages": rp.get("page_numbers", []),
            })
        result[qid] = {
            "answer": a.get("answer"),
            "refs": refs,
            "timing_ms": timing,
        }
    return result


def load_gold(path: str) -> dict:
    """Load gold_answers_final.json → {qid: gold_record}"""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["question_id"]: r for r in records}


def main():
    parser = argparse.ArgumentParser(description="ARLC local evaluator")
    parser.add_argument("--submission", required=True, help="Path to submission JSON")
    parser.add_argument(
        "--gold",
        default=str(ROOT_CHALLENGE / "gold_answers_final.json"),
        help="Path to gold answers JSON",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip LLM judge for free_text (score all as 0)"
    )
    parser.add_argument("--out", default=None, help="Save full results to JSON")
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel judge workers"
    )
    args = parser.parse_args()

    print(f"Loading submission: {args.submission}")
    pred_map = load_submission(args.submission)
    print(f"  {len(pred_map)} answers")

    print(f"Loading gold: {args.gold}")
    gold_map = load_gold(args.gold)
    print(f"  {len(gold_map)} gold answers\n")

    # Collect free_text items for judge
    asst_scores = {}
    if not args.no_judge:
        ft_items = []
        for qid, gold_rec in gold_map.items():
            if gold_rec.get("answer_type") == "free_text":
                pred_rec = pred_map.get(qid, {})
                ft_items.append({
                    "question_id": qid,
                    "question": gold_rec.get("question", ""),
                    "gold": str(gold_rec.get("answer", "")),
                    "pred": str(pred_rec.get("answer", "") or ""),
                })
        if ft_items:
            print(f"Running LLM judge on {len(ft_items)} free_text answers...")
            asst_scores = judge_batch(ft_items, workers=args.workers)
            ft_correct = sum(1 for s in asst_scores.values() if s == 1.0)
            print(f"  Judge done: {ft_correct}/{len(ft_items)} correct\n")
    else:
        print("Skipping LLM judge (--no-judge)\n")

    # Compute scores
    results = compute_scores(gold_map, pred_map, asst_scores)

    # Print report
    print("=" * 55)
    print(f"  ARLC EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Answers:    {results['n_answered']}/{results['n_total']}")
    print(f"  F (completeness):  {results['F']:.4f}")
    print(f"  T (timeliness):    {results['T']:.4f}")
    print(f"  G (grounding):     {results['G']:.4f}  [NOTE: local G=1.0 if refs match gold refs]")
    print(f"  S_det ({results['n_det']:3d} q):   {results['S_det']:.4f}")
    print(f"  S_asst({results['n_asst']:3d} q):  {results['S_asst']:.4f}")
    print("-" * 55)
    print(f"  TOTAL SCORE:       {results['total']:.4f}")
    print("=" * 55)

    if results.get("n_det"):
        det_details = [d for d in results["details"] if d["det_score"] is not None]
        det_wrong = [d for d in det_details if d["det_score"] < 1.0 and d["pred_answer"] is not None]
        print(f"\nDeterminate wrong ({len(det_wrong)}):")
        for d in sorted(det_wrong, key=lambda x: x["answer_type"])[:20]:
            print(f"  [{d['answer_type']:7s}] gold={repr(d['gold_answer'])[:30]:30s} "
                  f"pred={repr(d['pred_answer'])[:30]}")

    if args.out:
        # Save details (without full answer text to keep it small)
        out_data = {k: v for k, v in results.items() if k != "details"}
        out_data["details"] = results["details"]
        Path(args.out).write_text(json.dumps(out_data, ensure_ascii=False, indent=2))
        print(f"\nFull results saved to: {args.out}")


if __name__ == "__main__":
    main()
