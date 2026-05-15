"""End-to-end driver: Stage 1 → 6 for a single episode (or a season).

Usage:
  python -m benchmark.pipeline.run_pipeline single \
      --video data/raw/breaking_bad/dev/tos_4min_720p.mp4 \
      --series breaking_bad --episode tos_dev_4min

  python -m benchmark.pipeline.run_pipeline season \
      --series breaking_bad --episodes ep01 ep02 ep03
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.pipeline.common import CFG, get_logger  # noqa: E402

LOG = get_logger("driver")


def run_single(video: Path, series: str, episode: str) -> None:
    from benchmark.pipeline import stage1_perception, stage2_events, stage3_relations, stage4_trajectory

    LOG.info("=== Stage 1: perception ===")
    stage1_perception.run(video, series, episode)
    LOG.info("=== Stage 2: events ===")
    stage2_events.run(series, episode, skip_passB=False)
    LOG.info("=== Stage 3: M2 event DAG ===")
    stage3_relations.run(series, episode)
    LOG.info("=== Stage 4: M4 per-episode trajectory ===")
    stage4_trajectory.run(series, episode)
    LOG.info(f"=== Single-episode pipeline done for {series}/{episode} ===")


def run_season(series: str, episodes: list[str], tasks: list[str]) -> None:
    from benchmark.pipeline import stage4_5_cross_episode, stage4_6_semantic, stage5_qgen, stage6_filters

    LOG.info("=== Stage 4.5: cross-episode merge ===")
    stage4_5_cross_episode.run(series, episodes)
    LOG.info("=== Stage 4.6: semantic memory M5/M6/M7 ===")
    stage4_6_semantic.run(series, episodes)
    LOG.info("=== Stage 5: 9-task QA generation ===")
    stage5_qgen.run(series, episodes, tasks)
    LOG.info("=== Stage 6: quality filters ===")
    qa_in = Path(CFG["project"]["data_root"]) / "qa" / series / "qa_staging.jsonl"
    stage6_filters.run(series, qa_in)
    LOG.info(f"=== Season pipeline done for {series}: {episodes} ===")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p1 = sub.add_parser("single")
    p1.add_argument("--video", required=True, type=Path)
    p1.add_argument("--series", required=True)
    p1.add_argument("--episode", required=True)
    p2 = sub.add_parser("season")
    p2.add_argument("--series", required=True)
    p2.add_argument("--episodes", nargs="+", required=True)
    p2.add_argument("--tasks", nargs="+", default=["T1", "T2", "T4", "T5", "T6", "T7", "T8", "T9", "T10"])
    args = ap.parse_args()
    if args.mode == "single":
        run_single(args.video, args.series, args.episode)
    elif args.mode == "season":
        run_season(args.series, args.episodes, args.tasks)


if __name__ == "__main__":
    main()
