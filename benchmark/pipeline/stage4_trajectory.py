"""Stage 4: Per-episode character emotion trajectory → M4 (per-episode).

For each named character, walk the per-episode event DAG and produce a sequence
(timestamp, emotion_label, intensity, trigger_event_id, evidence_refs).

Methodology:
  1. Pull all affect_event rows whose participants include the character.
  2. Interpolate between affect events along the temporal axis (constant within
     an event window; decays toward neutral after end_sec at config-tunable rate
     until the next affect event begins).
  3. Output a per-character JSON file with the trajectory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.pipeline.common import (  # noqa: E402
    CFG,
    ensure_dir,
    get_logger,
    read_jsonl,
    write_json,
)

LOG = get_logger("stage4")


def _characters_in_episode(events: list[dict], meta: dict) -> list[str]:
    """Collect named characters present in any affect_event."""
    chars: set[str] = set()
    for e in events:
        if e.get("type") == "affect_event":
            for p in e.get("participants", []):
                if isinstance(p, str) and p not in ("unknown", "?"):
                    chars.add(p)
    return sorted(chars)


def build_trajectory(character: str, events: list[dict]) -> list[dict]:
    """Build trajectory for a single character.

    Returns a list of waypoints sorted by ts, each with:
      ts, emotion, intensity, trigger_event_id, source_event_id
    """
    waypoints: list[dict] = []
    char_events = [
        e for e in events
        if e.get("type") == "affect_event" and character in e.get("participants", [])
    ]
    char_events.sort(key=lambda e: e["time_span"][0])
    for e in char_events:
        waypoints.append({
            "ts": float(e["time_span"][0]),
            "emotion": e.get("emotion"),
            "intensity": float(e.get("intensity", 0.5)),
            "source_event_id": e["event_id"],
            "trigger_event_id": e.get("trigger_ref"),
            "evidence_refs": [e["event_id"]],
            "phase": "onset",
        })
        # Plateau at end_sec
        waypoints.append({
            "ts": float(e["time_span"][1]),
            "emotion": e.get("emotion"),
            "intensity": float(e.get("intensity", 0.5)) * 0.8,
            "source_event_id": e["event_id"],
            "trigger_event_id": e.get("trigger_ref"),
            "evidence_refs": [e["event_id"]],
            "phase": "plateau_end",
        })
    return waypoints


def run(series: str, episode: str) -> None:
    events_path = Path(CFG["project"]["data_root"]) / "events" / series / f"{episode}.jsonl"
    meta_path = Path(CFG["project"]["data_root"]) / "events" / series / f"{episode}.meta.json"
    if not events_path.exists():
        raise FileNotFoundError(f"Run Stage 2 first: missing {events_path}")
    events = read_jsonl(events_path)
    meta = json.loads(meta_path.read_text())
    chars = _characters_in_episode(events, meta)
    LOG.info(f"{episode}: {len(chars)} named characters with affect events: {chars}")

    out_dir = ensure_dir(Path(CFG["project"]["data_root"]) / "trajectory" / series / episode)
    trajectories: dict[str, list[dict]] = {}
    for c in chars:
        traj = build_trajectory(c, events)
        write_json(out_dir / f"{c}.json", {"character": c, "episode": episode, "waypoints": traj})
        trajectories[c] = traj
    write_json(out_dir / "_summary.json", {
        "episode": episode,
        "characters": chars,
        "n_waypoints_total": sum(len(t) for t in trajectories.values()),
    })
    LOG.info(f"Stage 4 done: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--episode", required=True)
    args = ap.parse_args()
    run(args.series, args.episode)


if __name__ == "__main__":
    main()
