"""Stage 4.6: Semantic memory distillation → M5 (OCEAN persona) + M6 (relations) + M7 (world).

Single long-context GPT-5.5 pass per memory type, taking all season events + M3 + per-char
trajectories as input.
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

LOG = get_logger("stage4_6")

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


def _gather_season_inputs(series: str, episodes: list[str]) -> dict:
    """Bundle events, M3, per-character M4 cumulative."""
    data_root = Path(CFG["project"]["data_root"])
    season_dir = data_root / "season" / series

    all_events: list[dict] = []
    for ep in episodes:
        events_path = data_root / "events" / series / f"{ep}.jsonl"
        if events_path.exists():
            all_events.extend(read_jsonl(events_path))

    M3 = {}
    if (season_dir / "M3.json").exists():
        M3 = json.loads((season_dir / "M3.json").read_text())

    M4_chars: dict[str, dict] = {}
    for f in season_dir.glob("M4_*.json"):
        if f.stem == "M4_index":
            continue
        ch = f.stem[len("M4_"):]
        M4_chars[ch] = json.loads(f.read_text())

    return {
        "events": all_events,
        "M3": M3,
        "M4_per_character": M4_chars,
        "characters": sorted(M4_chars.keys()),
    }


def distill_M5(bundle: dict, series: str, episodes: list[str]) -> dict:
    """Per-character OCEAN persona."""
    sys_prompt = (PROMPT_DIR / "semantic_M5_persona.md").read_text(encoding="utf-8")
    out: dict[str, dict] = {}
    for ch in bundle["characters"]:
        ch_events = [
            {"id": e["event_id"], "ts": e["time_span"], "type": e["type"],
             "emotion": e.get("emotion"), "intensity": e.get("intensity"),
             "summary": e.get("summary", "")[:300]}
            for e in bundle["events"] if ch in e.get("participants", [])
        ]
        user = (
            f"Character: {ch}\n"
            f"Episodes: {episodes}\n\n"
            f"Cumulative trajectory M4 (waypoints):\n"
            f"{json.dumps(bundle['M4_per_character'].get(ch, {}).get('waypoints', []), indent=2)[:20000]}\n\n"
            f"Events involving this character:\n"
            f"{json.dumps(ch_events, indent=2)[:30000]}\n\n"
            f"Cross-episode edges involving this character:\n"
            f"{json.dumps([e for e in bundle['M3'].get('cross_episode_edges', []) if ch in str(e)], indent=2)[:10000]}"
        )
        raw = llm_chat(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user},
            ],
            max_tokens=4000,
        )
        try:
            out[ch] = parse_json_block(raw)
        except Exception as e:  # noqa: BLE001
            LOG.warning(f"M5 distillation for {ch} failed: {e}")
            out[ch] = {"character": ch, "error": str(e)}
    return out


def distill_M6(bundle: dict, series: str) -> dict:
    """Relationship network."""
    sys_prompt = (PROMPT_DIR / "semantic_M6_relation.md").read_text(encoding="utf-8")
    user = (
        f"Series: {series}\nCharacters: {bundle['characters']}\n\n"
        f"All cross-episode edges:\n{json.dumps(bundle['M3'].get('cross_episode_edges', []), indent=2)[:30000]}\n\n"
        f"Social events sample (first 100):\n"
        f"{json.dumps([e for e in bundle['events'] if e.get('type') == 'social_event'][:100], indent=2)[:30000]}"
    )
    raw = llm_chat(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ],
        max_tokens=6000,
    )
    try:
        return parse_json_block(raw)
    except Exception as e:  # noqa: BLE001
        LOG.warning(f"M6 distillation failed: {e}")
        return {"error": str(e)}


def distill_M7(bundle: dict, series: str, M5: dict, M6: dict) -> dict:
    """Propositional plot state."""
    sys_prompt = (PROMPT_DIR / "semantic_M7_world.md").read_text(encoding="utf-8")
    # Provide event summaries + already-distilled M5/M6 for grounding
    event_summary = [
        {"id": e["event_id"], "ts": e["time_span"], "type": e["type"],
         "participants": e.get("participants", []), "summary": e.get("summary", "")[:200]}
        for e in bundle["events"]
    ]
    user = (
        f"Series: {series}\n"
        f"Personas (M5): {json.dumps({c: {'narrative_summary': p.get('narrative_summary', '')} for c, p in M5.items()}, indent=2)}\n"
        f"Relationships (M6) edges: {json.dumps(M6.get('edges', []), indent=2)[:10000]}\n\n"
        f"All events (compact):\n{json.dumps(event_summary, indent=2)[:40000]}"
    )
    raw = llm_chat(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ],
        max_tokens=6000,
    )
    try:
        return parse_json_block(raw)
    except Exception as e:  # noqa: BLE001
        LOG.warning(f"M7 distillation failed: {e}")
        return {"error": str(e)}


def run(series: str, episodes: list[str]) -> None:
    season_dir = ensure_dir(Path(CFG["project"]["data_root"]) / "season" / series)
    bundle = _gather_season_inputs(series, episodes)
    LOG.info(f"Loaded {len(bundle['events'])} events, {len(bundle['characters'])} characters")

    LOG.info("Distilling M5 personas...")
    M5 = distill_M5(bundle, series, episodes)
    write_json(season_dir / "M5_personas.json", M5)

    LOG.info("Distilling M6 relationships...")
    M6 = distill_M6(bundle, series)
    write_json(season_dir / "M6_relations.json", M6)

    LOG.info("Distilling M7 world state...")
    M7 = distill_M7(bundle, series, M5, M6)
    write_json(season_dir / "M7_world.json", M7)

    LOG.info(f"Stage 4.6 done: M5/M6/M7 under {season_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--episodes", nargs="+", required=True)
    args = ap.parse_args()
    run(args.series, args.episodes)


if __name__ == "__main__":
    main()
