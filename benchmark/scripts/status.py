"""Scan data/ subdirs and report per-series pipeline state.

Tells you, for each series:
  - Which episodes have which stages done
  - Which season-level outputs exist
  - How many QAs were generated and filtered
  - What's the recommended next step

Usage:
  python -m benchmark.scripts.status                  # all series
  python -m benchmark.scripts.status --series breaking_bad
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.pipeline.common import CFG  # noqa: E402


def _episodes_in(d: Path, ext: str = ".jsonl") -> set[str]:
    if not d.exists():
        return set()
    return {p.stem.replace(".meta", "") for p in d.glob(f"*{ext}") if not p.name.endswith(".meta.json")}


def report_series(series: str) -> None:
    data_root = Path(CFG["project"]["data_root"])
    perc = data_root / "perception" / series
    events = data_root / "events" / series
    eg = data_root / "event_graph" / series
    traj = data_root / "trajectory" / series
    season = data_root / "season" / series
    qa = data_root / "qa" / series
    final = data_root / "final" / series
    eval_d = data_root / "eval" / series

    # Per-episode stages
    s1 = _episodes_in(perc)
    s2 = _episodes_in(events)
    s3 = {p.stem.replace(".M2", "") for p in eg.glob("*.M2.json")} if eg.exists() else set()
    s4 = {d.name for d in traj.glob("*") if d.is_dir()} if traj.exists() else set()
    all_eps = s1 | s2 | s3 | s4

    print(f"\n# Pipeline status — series: {series}")
    print(f"\n## Per-episode")
    if not all_eps:
        print("  (no episodes yet)")
        return
    print(f"  {'episode':<20} S1  S2  S3  S4")
    for ep in sorted(all_eps):
        marks = [
            "✓ " if ep in s1 else "  ",
            "✓ " if ep in s2 else "  ",
            "✓ " if ep in s3 else "  ",
            "✓ " if ep in s4 else "  ",
        ]
        print(f"  {ep:<20} {' '.join(marks)}")

    # Season-level
    print(f"\n## Season-level")
    for k, fname in [
        ("M3 cross-ep DAG", "M3.json"),
        ("M5 personas (OCEAN)", "M5_personas.json"),
        ("M6 relations", "M6_relations.json"),
        ("M7 world state", "M7_world.json"),
    ]:
        path = season / fname
        if path.exists():
            data = json.loads(path.read_text())
            if "personas" in fname.lower():
                n = len(data) if isinstance(data, dict) else 0
                print(f"  ✓ {k}: {n} characters")
            elif "relations" in fname.lower():
                n = len(data.get("edges", []))
                print(f"  ✓ {k}: {n} edges")
            elif "world" in fname.lower():
                n = len(data.get("facts", []))
                print(f"  ✓ {k}: {n} propositional facts")
            else:
                n = len(data.get("cross_episode_edges", []))
                print(f"  ✓ {k}: {n} cross-ep edges")
        else:
            print(f"    {k}: missing")

    # QA + filters
    print(f"\n## QA")
    qa_staging = qa / "qa_staging.jsonl"
    qa_filtered = final / "qa_filtered.jsonl"
    if qa_staging.exists():
        n_gen = sum(1 for _ in open(qa_staging))
        print(f"  ✓ generated: {n_gen}")
        if qa_filtered.exists():
            n_kept = sum(1 for _ in open(qa_filtered))
            print(f"  ✓ filtered: {n_kept} kept ({n_gen - n_kept} dropped)")
        else:
            print(f"    filtered: not yet run")
    else:
        print(f"    not yet generated")

    # Eval
    print(f"\n## Eval summaries")
    if eval_d.exists():
        sums = sorted(eval_d.glob("*.summary.json"))
        if sums:
            for p in sums:
                d = json.loads(p.read_text())
                print(f"  ✓ {d['episode']} × {d['model']} × {d['setting']}: "
                      f"{d.get('overall_accuracy', 0)*100:.1f}% (N={d.get('n_questions')})")
        else:
            print(f"    none")
    else:
        print(f"    none")

    # Recommend next step
    print(f"\n## Suggested next action")
    if not s1:
        print(f"  → Run Stage 1 on a video. Example: make stage1 VIDEO=data/raw/{series}/X.mp4 EPISODE=ep01")
    elif s1 - s2:
        ep = sorted(s1 - s2)[0]
        print(f"  → Run Stage 2 on {ep}: python -m benchmark.pipeline.stage2_events --series {series} --episode {ep}")
    elif s2 - s3:
        ep = sorted(s2 - s3)[0]
        print(f"  → Run Stage 3 on {ep}: python -m benchmark.pipeline.stage3_relations --series {series} --episode {ep}")
    elif s3 - s4:
        ep = sorted(s3 - s4)[0]
        print(f"  → Run Stage 4 on {ep}: python -m benchmark.pipeline.stage4_trajectory --series {series} --episode {ep}")
    elif not (season / "M3.json").exists() and len(s4) >= 1:
        eps = " ".join(sorted(s4))
        print(f"  → Run Stage 4.5/4.6: make season SERIES={series} EPISODES='{eps}'")
    elif not qa_staging.exists():
        print(f"  → Run Stage 5: make season SERIES={series} EPISODES='{' '.join(sorted(s4))}'  # includes Stage 5+6")
    elif not qa_filtered.exists():
        print(f"  → Run Stage 6: python -m benchmark.pipeline.stage6_filters --series {series} --in_qa data/qa/{series}/qa_staging.jsonl")
    elif not eval_d.exists() or not list(eval_d.glob("*.summary.json")):
        ep = sorted(s4)[0] if s4 else "ep01"
        print(f"  → Run Stage 7: python -m benchmark.pipeline.stage7_eval --series {series} --episode {ep} --setting E0")
    else:
        print(f"  → Pipeline complete for this series. Try `make report SERIES={series}` for the cross-setting table.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default=None, help="default: scan all series under data/perception/")
    args = ap.parse_args()
    data_root = Path(CFG["project"]["data_root"])
    if args.series:
        report_series(args.series)
    else:
        perc_root = data_root / "perception"
        if not perc_root.exists():
            print("No data/perception/ directory — nothing to report.")
            return
        series_list = sorted(d.name for d in perc_root.iterdir() if d.is_dir())
        if not series_list:
            print("No series under data/perception/.")
            return
        for s in series_list:
            report_series(s)


if __name__ == "__main__":
    main()
