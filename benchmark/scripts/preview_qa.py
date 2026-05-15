"""Print surviving QAs after Stage 6 filters in a human-readable form.

Usage:
  python -m benchmark.scripts.preview_qa --series synthetic_demo
  python -m benchmark.scripts.preview_qa --series synthetic_demo --task T1
  python -m benchmark.scripts.preview_qa --series synthetic_demo --staged   # before filters

Shows: task, M-dependencies, question, all 4 options (★ marks correct),
the M-evidence the question was built from, and any filter signals retained.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.pipeline.common import CFG, read_jsonl  # noqa: E402


def _shorten(s, n: int = 160) -> str:
    if not isinstance(s, str):
        s = json.dumps(s, ensure_ascii=False, default=str)
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--task", default=None, help="filter to one task family (T1..T10)")
    ap.add_argument("--staged", action="store_true", help="preview qa_staging.jsonl (before filters)")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    data_root = Path(CFG["project"]["data_root"])
    src = data_root / ("qa" if args.staged else "final") / args.series / (
        "qa_staging.jsonl" if args.staged else "qa_filtered.jsonl"
    )
    if not src.exists():
        print(f"Not found: {src}")
        return

    rows = read_jsonl(src)
    if args.task:
        rows = [r for r in rows if r.get("task") == args.task]

    print(f"\n=== {src.relative_to(Path.cwd()) if src.is_absolute() else src} — N={len(rows)} ===")
    task_counts = Counter(r.get("task", "?") for r in rows)
    print("Per-task: " + ", ".join(f"{t}:{n}" for t, n in sorted(task_counts.items())))
    M_counts = Counter()
    for r in rows:
        ae = r.get("answer_evidence") or {}
        if isinstance(ae, dict):
            for m in ae.get("memory_components") or []:
                M_counts[m] += 1
    print("M-dependencies: " + ", ".join(f"{m}:{n}" for m, n in sorted(M_counts.items())))
    print()

    for i, r in enumerate(rows[: args.limit]):
        ae = r.get("answer_evidence") or {}
        mems = ae.get("memory_components") if isinstance(ae, dict) else None
        print(f"--- [{i + 1}] qid={r.get('qid')}  task={r.get('task')}  "
              f"M={mems}  horizon={r.get('horizon')} ---")
        if ae:
            print(f"  evidence: {_shorten(ae, 220)}")
        print(f"  Q: {_shorten(r['question'], 300)}")
        opts = r.get("options") or {}
        gold = r.get("correct") or r.get("answer") or r.get("gold")
        for k in ("A", "B", "C", "D"):
            mark = "★" if k == gold else " "
            print(f"   {mark}{k}. {_shorten(opts.get(k, ''), 180)}")
        filt = r.get("filter_outcomes") or r.get("filter_signals")
        if filt:
            print(f"  filter: {filt}")
        print()

    if len(rows) > args.limit:
        print(f"... ({len(rows) - args.limit} more not shown; use --limit to expand)")


if __name__ == "__main__":
    main()
