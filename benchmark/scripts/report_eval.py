"""Compare all Stage 7 eval summary.json files for a series.

Reads data/eval/<series>/*.summary.json and prints a unified table per (model, setting).

Usage:
  python -m benchmark.scripts.report_eval --series synthetic_demo
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.pipeline.common import CFG  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    args = ap.parse_args()

    base = Path(CFG["project"]["data_root"]) / "eval" / args.series
    if not base.exists():
        print(f"No eval directory: {base}")
        return

    # Group by (episode, model)
    by_em: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for p in sorted(base.glob("*.summary.json")):
        d = json.loads(p.read_text())
        ep, model = d["episode"], d["model"]
        setting = d["setting"]
        by_em[(ep, model)][setting] = d

    if not by_em:
        print(f"No summary.json files under {base}")
        return

    print(f"\n# Eval report — series: {args.series}\n")
    for (ep, model), settings in by_em.items():
        print(f"\n## Episode `{ep}` × Model `{model}`")
        tasks_seen = sorted({t for d in settings.values() for t in d.get("per_task_accuracy", {})})
        header = f"{'setting':>8} | {'N':>3} | {'overall':>8} | " + " ".join(f"{t:>5}" for t in tasks_seen)
        print(header)
        print("-" * len(header))
        for s in ["E0", "E1", "E2", "E3"]:
            d = settings.get(s)
            if not d:
                continue
            pt = d.get("per_task_accuracy", {})
            row = f"{s:>8} | {d.get('n_questions', 0):>3} | {d.get('overall_accuracy', 0)*100:>7.1f}% | "
            row += " ".join(f"{pt.get(t, 0)*100:>4.0f}%" for t in tasks_seen)
            print(row)

        # Gap analysis
        accs = {s: settings[s]["overall_accuracy"] for s in ["E0", "E1", "E2", "E3"] if s in settings}
        if "E0" in accs and "E1" in accs:
            print(f"\n  Acc(E1)-Acc(E0) = {(accs['E1']-accs['E0'])*100:+.1f}pp  (local→episode gain)")
        if "E1" in accs and "E2" in accs:
            print(f"  Acc(E2)-Acc(E1) = {(accs['E2']-accs['E1'])*100:+.1f}pp  (episode→season gain)")
        if "E2" in accs and "E3" in accs:
            print(f"  Acc(E3)-Acc(E2) = {(accs['E3']-accs['E2'])*100:+.1f}pp  (semantic-memory gain)")


if __name__ == "__main__":
    main()
