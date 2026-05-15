"""Stage 4.5: Cross-episode merge → M3 (cross-episode DAG) + M4 cumulative trajectory.

Takes per-episode M2 graphs and per-episode M4 trajectories, merges into season-level
M3 and per-character cumulative M4.

Cross-episode edge types: char_continuity, plot_continuity, relation_evolution, long_causal.
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
    llm_chat,
    parse_json_block,
    read_jsonl,
    write_json,
)

LOG = get_logger("stage4_5")


def _episode_summary(series: str, episode: str) -> dict:
    """Compact episode summary for cross-ep LLM input."""
    events = read_jsonl(Path(CFG["project"]["data_root"]) / "events" / series / f"{episode}.jsonl")
    M2 = json.loads((Path(CFG["project"]["data_root"]) / "event_graph" / series / f"{episode}.M2.json").read_text())
    return {
        "episode": episode,
        "n_events": len(events),
        "events": [
            {
                "id": e["event_id"],
                "ts": e["time_span"],
                "type": e["type"],
                "participants": e.get("participants", []),
                "emotion": e.get("emotion"),
                "intensity": e.get("intensity"),
                "summary": e.get("summary", "")[:200],
            }
            for e in events
        ],
        "high_conf_edges": M2.get("high_confidence_edges", []),
    }


def infer_cross_ep_edges(summaries: list[dict]) -> list[dict]:
    """Use GPT-5.5 long-ctx to propose cross-episode edges across the season."""
    sys_prompt = (
        "You are constructing the cross-episode event graph M3 for a TV series season. "
        "Given per-episode event summaries, propose edges that span across episodes. "
        "Edge types: char_continuity (same character carrying emotional state across episodes), "
        "plot_continuity (plot consequence playing out later), relation_evolution (a relationship "
        "changes between two characters across episodes), long_causal (clear cause-effect across "
        "episodes), predicts_action_cross_ep (affect in episode N predicts action in episode N+k).\n"
        "Output JSON: {edges: [{src_event_id, dst_event_id, type, confidence, evidence}, ...]}. "
        "Be conservative; only propose edges with strong evidence."
    )
    user = "Episode summaries:\n" + json.dumps(summaries, indent=2)[:60000]
    raw = llm_chat(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ],
        max_tokens=8000,
    )
    try:
        obj = parse_json_block(raw)
        return obj.get("edges", obj if isinstance(obj, list) else [])
    except Exception as e:  # noqa: BLE001
        LOG.warning(f"cross-ep edge parse failed: {e}")
        return []


def merge_trajectories(series: str, episodes: list[str]) -> dict[str, list[dict]]:
    """Concatenate per-episode M4 trajectories per character."""
    base = Path(CFG["project"]["data_root"]) / "trajectory" / series
    out: dict[str, list[dict]] = {}
    for ep in episodes:
        ep_dir = base / ep
        if not ep_dir.exists():
            LOG.warning(f"Missing per-episode trajectory: {ep_dir}")
            continue
        for char_file in ep_dir.glob("*.json"):
            if char_file.stem == "_summary":
                continue
            data = json.loads(char_file.read_text())
            ch = data["character"]
            waypoints = [
                {**w, "episode": ep, "season_ts": None}  # season_ts to be assigned by caller
                for w in data.get("waypoints", [])
            ]
            out.setdefault(ch, []).extend(waypoints)
    # Sort each character's trajectory by (episode, ts)
    ep_order = {ep: i for i, ep in enumerate(episodes)}
    for ch, wps in out.items():
        wps.sort(key=lambda w: (ep_order.get(w["episode"], 999), w["ts"]))
    return out


def run(series: str, episodes: list[str]) -> None:
    LOG.info(f"Merging {len(episodes)} episodes for {series}: {episodes}")
    summaries = []
    for ep in episodes:
        try:
            summaries.append(_episode_summary(series, ep))
        except FileNotFoundError as e:
            LOG.warning(f"Skipping {ep}: {e}")
    if not summaries:
        LOG.error("No usable episode summaries; abort.")
        return

    cross_edges = infer_cross_ep_edges(summaries)
    LOG.info(f"Inferred {len(cross_edges)} cross-episode edges")

    season_dir = ensure_dir(Path(CFG["project"]["data_root"]) / "season" / series)
    M3 = {
        "series": series,
        "episodes": episodes,
        "cross_episode_edges": cross_edges,
        "per_episode_event_counts": {ep: s["n_events"] for ep, s in zip(episodes, summaries)},
    }
    write_json(season_dir / "M3.json", M3)
    LOG.info(f"Wrote {season_dir/'M3.json'}")

    # M4 cumulative
    M4 = merge_trajectories(series, episodes)
    for ch, wps in M4.items():
        write_json(season_dir / f"M4_{ch}.json", {"character": ch, "season": series, "waypoints": wps})
    write_json(season_dir / "M4_index.json", {
        "characters": sorted(M4.keys()),
        "n_waypoints": {c: len(wps) for c, wps in M4.items()},
    })
    LOG.info(f"Stage 4.5 done: M3 + {len(M4)} character trajectories")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--episodes", nargs="+", required=True)
    args = ap.parse_args()
    run(args.series, args.episodes)


if __name__ == "__main__":
    main()
