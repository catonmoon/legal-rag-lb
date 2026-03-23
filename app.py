"""
app.py — ARLC local evaluator (Gradio UI)

Upload submission_final_v1.json → see scores + per-question breakdown.
"""

import json
import os
import sys
from pathlib import Path

import gradio as gr
import pandas as pd
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorer import compute_scores
from judge import judge_batch

_eval_dir = Path(__file__).resolve().parent
_starter_kit = _eval_dir.parent / "starter_kit"

def _find(env_var, *candidates):
    if env_var in os.environ:
        return Path(os.environ[env_var])
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return Path(candidates[-1])

GOLD_PATH      = _find("GOLD_PATH",      _eval_dir / "gold_answers_final.json",      _starter_kit / "gold_answers_final.json")
QUESTIONS_PATH = _find("QUESTIONS_PATH", _eval_dir / "questions_final.json",          _starter_kit / "questions_final.json")
DOCS_DIR       = _find("DOCS_DIR",       _eval_dir / "docs_final_md_fast",            _starter_kit / "docs_final_md_fast")
PDF_DIR        = _find("PDF_DIR",        _eval_dir / "docs_final",                    _starter_kit / "docs_final")

# URL prefix for static file links (e.g. "/eval" when behind a reverse proxy)
_root = os.getenv("ROOT_PATH", "").rstrip("/")
# Strip scheme if ROOT_PATH is a full URL (e.g. https://manzherok.ru/eval → /eval)
if _root.startswith("http"):
    from urllib.parse import urlparse
    _root = urlparse(_root).path.rstrip("/")
STATIC_PREFIX = _root  # e.g. "" locally, "/eval" on server

def _load_questions() -> dict:
    try:
        qs = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
        return {q["id"]: q["question"] for q in qs}
    except Exception:
        return {}

_Q_MAP: dict = {}  # loaded once on first evaluate

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_gold(path) -> dict:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return {r["question_id"]: r for r in records}


def load_submission(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    answers = data if isinstance(data, list) else data.get("answers", [])
    result = {}
    for a in answers:
        qid = a.get("question_id", a.get("id", ""))
        tel = a.get("telemetry", {})
        timing = tel.get("timing", {}).get("total_time_ms", 0)
        raw_pages = tel.get("retrieval", {}).get("retrieved_chunk_pages", [])
        refs = [{"doc_id": rp.get("doc_id", ""), "pages": rp.get("page_numbers", [])} for rp in raw_pages]
        result[qid] = {"answer": a.get("answer"), "refs": refs, "timing_ms": timing}
    return result


# ── HTML formatters ───────────────────────────────────────────────────────────

def _score_badge(s) -> str:
    if s is None:
        return '<span style="color:#999">—</span>'
    if s >= 1.0:
        return '<span style="color:green;font-weight:bold">✓</span>'
    if s > 0:
        return f'<span style="color:orange">{s:.2f}</span>'
    return '<span style="color:red;font-weight:bold">✗</span>'


def _fmt(v) -> str:
    """Format value: None → null, lists/strings as JSON."""
    if v is None:
        return "null"
    if isinstance(v, list):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _get_page_text(doc_id: str, page_num: int) -> str:
    """Extract a single page from a doc MD file."""
    f = DOCS_DIR / f"{doc_id}.md"
    if not f.exists():
        return f"(doc not found: {doc_id[:20]}…)"
    text = f.read_text(encoding="utf-8")
    pages = text.split("<!-- page:")
    for pg in pages[1:]:
        try:
            num = int(pg.split("-->")[0].strip())
        except ValueError:
            continue
        if num == page_num:
            return pg[pg.find("-->") + 3:].strip()[:3000]
    return f"(page {page_num} not found)"


def _refs_html(refs: list, label: str = "") -> str:
    if not refs:
        return '<span style="color:#bbb">—</span>'
    parts = []
    seen = set()
    for r in refs:
        doc_id = r.get("doc_id", "")
        pages = r.get("pages", r.get("page_numbers", []))
        doc_short = doc_id[:12] + "…"

        # Links to open full doc in browser (served via /static_md/ and /static_pdf/)
        md_path = DOCS_DIR / f"{doc_id}.md"
        pdf_path = PDF_DIR / f"{doc_id}.pdf"
        open_links = ""
        if md_path.exists():
            open_links += f' <a href="{STATIC_PREFIX}/static_md/{doc_id}.md" target="_blank" style="font-size:10px;color:#666">[MD]</a>'
        if pdf_path.exists():
            open_links += f' <a href="{STATIC_PREFIX}/static_pdf/{doc_id}.pdf" target="_blank" style="font-size:10px;color:#666">[PDF]</a>'

        for pg in pages:
            key = (doc_id, pg)
            if key in seen:
                continue
            seen.add(key)
            page_text = _get_page_text(doc_id, pg)
            escaped = page_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            parts.append(
                f'<details style="display:inline-block;margin:2px;vertical-align:top">'
                f'<summary style="cursor:pointer;color:#1565c0;font-size:11px">'
                f'{doc_short} p.{pg}{open_links}</summary>'
                f'<pre style="background:#fff;border:1px solid #ddd;border-radius:4px;'
                f'padding:8px;font-size:11px;max-height:300px;overflow-y:auto;'
                f'white-space:pre-wrap;word-break:break-word;min-width:400px;max-width:700px">'
                f'{escaped}</pre></details>'
            )
    return "\n".join(parts) if parts else '<span style="color:#bbb">—</span>'


def build_details_html(details: list, filter_type: str = "all", filter_score: str = "all") -> str:
    if not details:
        return "<p>No details.</p>"

    rows = details
    if filter_type != "all":
        rows = [d for d in rows if d.get("answer_type") == filter_type]
    if filter_score == "wrong":
        rows = [d for d in rows if (d.get("det_score") is not None and d["det_score"] < 1.0)
                or (d.get("det_score") is None and d.get("score", 1) < 1.0)]
    elif filter_score == "correct":
        rows = [d for d in rows if (d.get("det_score") is not None and d["det_score"] >= 1.0)
                or (d.get("det_score") is None and d.get("score", 0) >= 1.0)]

    if not rows:
        return "<p>No items match filter.</p>"

    html = f'<p style="color:#666;font-size:13px">Showing {len(rows)} questions</p>'
    for d in rows:
        qid = d.get("question_id", "")
        atype = d.get("answer_type", "")
        gold = d.get("gold_answer")
        pred = d.get("pred_answer")
        gold_refs = d.get("gold_refs", [])
        pred_refs = d.get("pred_refs", [])
        score = d.get("score", 0)
        det_score = d.get("det_score")
        grounding = d.get("grounding", 0)
        timing_ok = d.get("timing_ok", 1)

        question_text = _Q_MAP.get(qid, qid[:24] + "…")

        if det_score is not None:
            is_correct = det_score >= 1.0
        else:
            is_correct = score >= 1.0

        bg = "#eaffea" if is_correct else "#ffeaea"
        border = "green" if is_correct else "red"
        icon = "✓" if is_correct else "✗"

        html += f"""
<div style="background:{bg};border-radius:6px;padding:10px 14px;margin-bottom:8px;font-size:13px;border-left:4px solid {border}">
  <div style="display:flex;justify-content:space-between;margin-bottom:6px">
    <span><code style="background:#0001;padding:1px 4px;border-radius:3px;font-size:11px">{atype}</code>
    &nbsp;<b>{question_text}</b></span>
    <span style="white-space:nowrap;margin-left:12px">{icon} &nbsp; G={grounding:.2f} T={'✓' if timing_ok else '✗'}</span>
  </div>
  <div style="margin-bottom:2px"><b>Gold:</b> <code>{_fmt(gold)}</code></div>
  <div style="margin-bottom:2px"><b>Pred:</b> <code>{_fmt(pred)}</code></div>
  <div style="font-size:11px;color:#666;margin-top:4px">
    <b>Gold refs:</b> {_refs_html(gold_refs)} &nbsp;|&nbsp;
    <b>Pred refs:</b> {_refs_html(pred_refs)}
  </div>
</div>"""

    return html


def _type_table_html(details: list) -> str:
    from collections import defaultdict
    by_type = defaultdict(lambda: {"total": 0, "correct": 0, "wrong": 0})
    for d in details:
        atype = d.get("answer_type", "?")
        by_type[atype]["total"] += 1
        det = d.get("det_score")
        score = d.get("score", 0)
        is_ok = (det >= 1.0) if det is not None else (score >= 1.0)
        if is_ok:
            by_type[atype]["correct"] += 1
        else:
            by_type[atype]["wrong"] += 1

    order = ["boolean", "date", "number", "name", "names", "free_text"]
    rows_html = ""
    for atype in order:
        if atype not in by_type:
            continue
        t = by_type[atype]
        acc = t["correct"] / t["total"] if t["total"] else 0
        bar_w = int(acc * 80)
        bar_color = "#4caf50" if acc >= 0.97 else ("#ff9800" if acc >= 0.90 else "#f44336")
        rows_html += f"""
<tr style="border-bottom:1px solid #eee">
  <td style="padding:4px 10px;font-weight:bold">{atype}</td>
  <td style="padding:4px 10px;text-align:right">{t['total']}</td>
  <td style="padding:4px 10px;text-align:right;color:green">{t['correct']}</td>
  <td style="padding:4px 10px;text-align:right;color:{'red' if t['wrong'] else '#999'}">{t['wrong']}</td>
  <td style="padding:4px 10px">
    <div style="display:inline-block;width:{bar_w}px;height:12px;background:{bar_color};border-radius:2px;vertical-align:middle"></div>
    <span style="margin-left:6px;font-size:12px">{acc:.1%}</span>
  </td>
</tr>"""

    total_q = sum(t["total"] for t in by_type.values())
    total_c = sum(t["correct"] for t in by_type.values())
    total_w = sum(t["wrong"] for t in by_type.values())
    return f"""
<table style="font-family:monospace;font-size:13px;border-collapse:collapse;margin-top:12px;width:100%">
  <thead>
    <tr style="background:#f0f0f0">
      <th style="padding:6px 10px;text-align:left">Type</th>
      <th style="padding:6px 10px;text-align:right">Total</th>
      <th style="padding:6px 10px;text-align:right">Correct</th>
      <th style="padding:6px 10px;text-align:right">Wrong</th>
      <th style="padding:6px 10px;text-align:left">Accuracy</th>
    </tr>
  </thead>
  <tbody>{rows_html}
    <tr style="background:#f8f8f8;font-weight:bold">
      <td style="padding:6px 10px">TOTAL</td>
      <td style="padding:6px 10px;text-align:right">{total_q}</td>
      <td style="padding:6px 10px;text-align:right;color:green">{total_c}</td>
      <td style="padding:6px 10px;text-align:right;color:{'red' if total_w else '#999'}">{total_w}</td>
      <td style="padding:6px 10px">{total_c/total_q:.1%}</td>
    </tr>
  </tbody>
</table>"""


def build_summary_html(results: dict) -> str:
    total = results["total"]
    color = "green" if total >= 0.85 else ("orange" if total >= 0.7 else "red")
    type_table = _type_table_html(results.get("details", []))
    return f"""
<div style="font-family:monospace;background:#f8f8f8;border-radius:8px;padding:16px 20px;font-size:14px">
  <div style="font-size:24px;font-weight:bold;color:{color};margin-bottom:12px">
    TOTAL: {total:.4f}
  </div>
  <table>
    <tr><td>S_det</td><td style="padding-left:16px"><b>{results['S_det']:.4f}</b></td>
        <td style="padding-left:24px;color:#666">{results['n_det']} determinate questions</td></tr>
    <tr><td>S_asst</td><td style="padding-left:16px"><b>{results['S_asst']:.4f}</b></td>
        <td style="padding-left:24px;color:#666">{results['n_asst']} free_text questions</td></tr>
    <tr><td>G</td><td style="padding-left:16px"><b>{results['G']:.4f}</b></td>
        <td style="padding-left:24px;color:#666">grounding (F-beta β=2.5)</td></tr>
    <tr><td>T</td><td style="padding-left:16px"><b>{results['T']:.4f}</b></td>
        <td style="padding-left:24px;color:#666">telemetry validity (≤100s)</td></tr>
    <tr><td>F</td><td style="padding-left:16px"><b>{results['F']:.4f}</b></td>
        <td style="padding-left:24px;color:#666">TTFT factor — mean latency: <b>{results['mean_ttft_ms']:,} ms</b> &nbsp;({results['n_answered']}/{results['n_total']} answered)</td></tr>
  </table>
  <div style="margin-top:10px;color:#888;font-size:12px">
    Formula: (0.7 × S_det + 0.3 × S_asst) × G × T × F
  </div>
  {type_table}
</div>"""


# ── State ─────────────────────────────────────────────────────────────────────

_last_details: list = []
_last_results: dict = {}


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(file_obj, use_judge: bool, workers: int):
    global _last_details, _last_results, _Q_MAP

    if file_obj is None:
        return "❌ Upload a submission JSON first", "", build_details_html([])

    if not _Q_MAP:
        _Q_MAP.update(_load_questions())

    gold_map = load_gold(GOLD_PATH)
    pred_map = load_submission(file_obj.name)

    asst_scores = {}
    judge_msg = ""
    if use_judge:
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
            asst_scores = judge_batch(ft_items, workers=workers)
            ft_correct = sum(1 for s in asst_scores.values() if s == 1.0)
            judge_msg = f" | Judge: {ft_correct}/{len(ft_items)} correct"
    else:
        judge_msg = " | Judge: skipped"

    results = compute_scores(gold_map, pred_map, asst_scores)

    # Enrich details with question text + refs
    for d in results["details"]:
        qid = d["question_id"]
        d["gold_refs"] = gold_map.get(qid, {}).get("refs", [])
        d["pred_refs"] = pred_map.get(qid, {}).get("refs", [])

    _last_details = results["details"]
    _last_results = results

    status = f"✅ Evaluated {results['n_answered']}/{results['n_total']} answers{judge_msg}"
    summary = build_summary_html(results)
    details_html = build_details_html(_last_details, "all", "wrong")
    return status, summary, details_html


def refresh_details(filter_type: str, filter_score: str):
    return build_details_html(_last_details, filter_type, filter_score)


def build_wrong_table():
    if not _last_details:
        return pd.DataFrame()
    rows = []
    for d in _last_details:
        if d.get("det_score") is not None and d["det_score"] < 1.0:
            rows.append({
                "qid": d["question_id"][:16] + "…",
                "type": d["answer_type"],
                "gold": _fmt(d["gold_answer"])[:50],
                "pred": _fmt(d["pred_answer"])[:50],
                "G": round(d["grounding"], 3),
            })
    if not rows:
        return pd.DataFrame(columns=["qid", "type", "gold", "pred", "G"])
    return pd.DataFrame(rows)


# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="ARLC Evaluator") as demo:
        gr.Markdown("# ARLC Local Evaluator\nUpload `submission_final_v1.json` → see scores + error breakdown.")

        with gr.Row():
            file_in = gr.File(label="Submission JSON", file_types=[".json"])
            with gr.Column():
                use_judge = gr.Checkbox(label="Run LLM judge (free_text)", value=True)
                workers = gr.Slider(1, 16, value=8, step=1, label="Judge workers")
                eval_btn = gr.Button("Evaluate", variant="primary")

        status_out = gr.Markdown()
        summary_out = gr.HTML()

        gr.Markdown("---")
        gr.Markdown("## Wrong determinate answers")
        wrong_table = gr.Dataframe(interactive=False, wrap=True)

        gr.Markdown("---")
        gr.Markdown("## Per-question details")
        with gr.Row():
            filter_type = gr.Dropdown(
                choices=["all", "boolean", "date", "number", "name", "names", "free_text"],
                value="all", label="Answer type"
            )
            filter_score = gr.Dropdown(
                choices=["all", "wrong", "correct"],
                value="wrong", label="Score filter"
            )
            filter_btn = gr.Button("Filter")

        details_out = gr.HTML()

        # Events
        eval_btn.click(
            fn=evaluate,
            inputs=[file_in, use_judge, workers],
            outputs=[status_out, summary_out, details_out],
        ).then(fn=build_wrong_table, inputs=[], outputs=[wrong_table])

        filter_btn.click(
            fn=refresh_details,
            inputs=[filter_type, filter_score],
            outputs=[details_out],
        )

    return demo


if __name__ == "__main__":
    import time
    root_path = os.getenv("ROOT_PATH", "")
    demo = build_ui()
    fastapi_app, _, _ = demo.launch(
        prevent_thread_lock=True,
        server_name="0.0.0.0",
        server_port=7861,
        root_path=root_path,
        theme=gr.themes.Soft(),
    )
    static_md_path  = "/static_md"
    static_pdf_path = "/static_pdf"
    if DOCS_DIR.exists():
        fastapi_app.mount(static_md_path,  StaticFiles(directory=str(DOCS_DIR)), name="docs_md")
        print(f"Static MD  → {DOCS_DIR}")
    else:
        print(f"Static MD  skipped (dir not found: {DOCS_DIR})")
    if PDF_DIR.exists():
        fastapi_app.mount(static_pdf_path, StaticFiles(directory=str(PDF_DIR)),  name="docs_pdf")
        print(f"Static PDF → {PDF_DIR}")
    else:
        print(f"Static PDF skipped (dir not found: {PDF_DIR})")
    print(f"Static MD  → {DOCS_DIR}")
    print(f"Static PDF → {PDF_DIR}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
