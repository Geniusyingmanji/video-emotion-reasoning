"""Stage 5: 9-task QA generation.

For each (task family, seed event/relation/etc.), build the per-task prompt context,
call GPT-5.5 with the task-specific prompt, parse the JSON output, validate, and
write to the QA staging file.

Tasks: T1 T2 T4 T5 T6 T7 T8 T9 T10

Run:
  python stage5_qgen.py --series breaking_bad --episodes tos_dev_4min --tasks T1 T2 T5
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
    write_jsonl,
)

LOG = get_logger("stage5")

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


# -----------------------------------------------------------------------------
# Task seed selection
# -----------------------------------------------------------------------------
def seed_T1(events: list[dict], _ctx: dict) -> list[dict]:
    """T1 seeds: any affect_event with named participant and intensity >= 0.4."""
    return [
        {"seed_event": e, "_kind": "T1"}
        for e in events
        if e.get("type") == "affect_event"
        and e.get("intensity", 0.0) >= 0.4
        and any(p not in ("unknown", "?") for p in e.get("participants", []))
    ]


def seed_T2(events: list[dict], _ctx: dict) -> list[dict]:
    """T2 seeds: affect_events with ≥2 cues across ≥2 modalities."""
    out = []
    for e in events:
        if e.get("type") != "affect_event":
            continue
        cues = e.get("cues", [])
        mods = {c.get("modality") for c in cues}
        if len(cues) >= 2 and len(mods) >= 2:
            out.append({"seed_event": e, "_kind": "T2"})
    return out


def seed_T4(events: list[dict], _ctx: dict) -> list[dict]:
    """T4 seeds: pairs of adjacent affect_events for the same character with different emotion."""
    per_char_events: dict[str, list[dict]] = {}
    for e in events:
        if e.get("type") == "affect_event":
            for p in e.get("participants", []):
                per_char_events.setdefault(p, []).append(e)
    out = []
    for c, evs in per_char_events.items():
        evs = sorted(evs, key=lambda x: x["time_span"][0])
        for a, b in zip(evs, evs[1:]):
            if a.get("emotion") != b.get("emotion"):
                out.append({"evA": a, "evB": b, "character": c, "_kind": "T4"})
    return out


def seed_T5(events: list[dict], ctx: dict) -> list[dict]:
    """T5 seeds: affect_events with a non-null trigger_ref (single-hop cause)."""
    id_to_event = {e["event_id"]: e for e in events}
    out = []
    for e in events:
        if e.get("type") != "affect_event":
            continue
        trig = e.get("trigger_ref")
        if trig and trig in id_to_event:
            out.append({"seed_event": e, "trigger_event": id_to_event[trig], "_kind": "T5"})
    return out


def seed_T6(events: list[dict], ctx: dict) -> list[dict]:
    """T6 seeds: affect_events with a causal chain of ≥3 hops in M2 high_confidence_edges."""
    id_to_event = {e["event_id"]: e for e in events}
    M2 = ctx.get("M2", {})
    edges = M2.get("high_confidence_edges", [])
    # Build adjacency: dst → src
    from collections import defaultdict

    parents: dict[str, list[str]] = defaultdict(list)
    for ed in edges:
        if ed.get("type") in ("causal", "emotion_trigger", "predicts_action"):
            parents[ed["dst"]].append(ed["src"])

    def trace_chain(node: str, depth: int, visited: set[str]) -> list[str] | None:
        if depth == 0:
            return [node]
        for p in parents.get(node, []):
            if p in visited:
                continue
            sub = trace_chain(p, depth - 1, visited | {p})
            if sub is not None:
                return [node] + sub
        return None

    out = []
    for e in events:
        if e.get("type") != "affect_event":
            continue
        chain = trace_chain(e["event_id"], 3, set())
        if chain and len(chain) >= 3:
            out.append({"seed_event": e, "chain": list(reversed(chain)), "_kind": "T6"})
    return out


def seed_T7(events: list[dict], ctx: dict) -> list[dict]:
    """T7 seeds: affect_event followed by action_event for the same character within 1min or 5min."""
    out = []
    affect = [e for e in events if e.get("type") == "affect_event"]
    actions = [e for e in events if e.get("type") == "action_event"]
    for a in affect:
        a_end = a["time_span"][1]
        for ac in actions:
            ac_start = ac["time_span"][0]
            if ac_start <= a_end:
                continue
            dt = ac_start - a_end
            if not (set(a.get("participants", [])) & set(ac.get("participants", []))):
                continue
            for horizon, max_dt in (("1min", 60), ("5min", 300)):
                if dt <= max_dt:
                    out.append({"seed_event": a, "predicted_action": ac, "horizon": horizon, "_kind": "T7"})
                    break
    return out


def seed_T8(_events: list[dict], ctx: dict) -> list[dict]:
    """T8 seeds: cross-episode edges of type predicts_action_cross_ep or long_causal in M3."""
    M3 = ctx.get("M3", {})
    out = []
    for ed in M3.get("cross_episode_edges", []):
        if ed.get("type") in ("predicts_action_cross_ep", "long_causal"):
            out.append({"cross_edge": ed, "_kind": "T8"})
    return out


def seed_T9(_events: list[dict], ctx: dict) -> list[dict]:
    """T9 seeds: each character with a complete M5 persona × one action_event used as the situation."""
    M5 = ctx.get("M5", {})
    out = []
    for char, persona in M5.items():
        if not isinstance(persona, dict) or "ocean" not in persona:
            continue
        actions = [e for e in ctx["events"] if e.get("type") == "action_event" and char in e.get("participants", [])]
        for ac in actions[:5]:  # up to 5 per character
            out.append({"character": char, "M5": persona, "action_event": ac, "_kind": "T9"})
    return out


def seed_T10(events: list[dict], ctx: dict) -> list[dict]:
    """T10 seeds: pairs (source_event, target_char) where M6 has an edge between source's char and target."""
    M6 = ctx.get("M6", {})
    out = []
    for src in events:
        if src.get("type") != "affect_event":
            continue
        for p in src.get("participants", []):
            for ed in M6.get("edges", []):
                src_name = ed.get("src")
                dst_name = ed.get("dst")
                if src_name == p:
                    # Find target's response within +5min
                    responses = [
                        e for e in events
                        if dst_name in e.get("participants", [])
                        and e["time_span"][0] >= src["time_span"][1]
                        and e["time_span"][0] - src["time_span"][1] <= 300
                    ]
                    if responses:
                        out.append({
                            "source_event": src,
                            "source_char": p,
                            "target_char": dst_name,
                            "relation": ed,
                            "response_events": responses[:3],
                            "_kind": "T10",
                        })
    return out


SEED_FNS = {
    "T1": seed_T1, "T2": seed_T2, "T4": seed_T4, "T5": seed_T5,
    "T6": seed_T6, "T7": seed_T7, "T8": seed_T8, "T9": seed_T9, "T10": seed_T10,
}


# -----------------------------------------------------------------------------
# QA generation
# -----------------------------------------------------------------------------
def _format_seed_for_prompt(seed: dict) -> str:
    """Render seed dict as readable text for the LLM."""
    return json.dumps(seed, indent=2, default=str)[:8000]


def generate_qa_for_seed(task: str, seed: dict, episode: str) -> dict | None:
    sys_prompt = (PROMPT_DIR / f"qgen_{task}.md").read_text(encoding="utf-8")
    user = f"Episode: {episode}\nTask: {task}\nSeed:\n{_format_seed_for_prompt(seed)}"
    try:
        raw = llm_chat(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user},
            ],
            max_tokens=2000,
        )
        qa = parse_json_block(raw)
        # Basic validation
        for k in ("task", "qid", "question", "options", "correct"):
            if k not in qa:
                LOG.warning(f"qa missing {k}: {qa}")
                return None
        if set(qa["options"].keys()) != {"A", "B", "C", "D"}:
            LOG.warning(f"qa options must be A/B/C/D: {qa.get('options')}")
            return None
        if qa["correct"] not in {"A", "B", "C", "D"}:
            return None
        return qa
    except Exception as e:  # noqa: BLE001
        LOG.warning(f"QA gen failed ({task}): {e}")
        return None


def _load_episode_ctx(series: str, episode: str) -> dict:
    data_root = Path(CFG["project"]["data_root"])
    events = read_jsonl(data_root / "events" / series / f"{episode}.jsonl")
    M2_path = data_root / "event_graph" / series / f"{episode}.M2.json"
    M2 = json.loads(M2_path.read_text()) if M2_path.exists() else {}
    return {"events": events, "M2": M2}


def _load_season_ctx(series: str) -> dict:
    season_dir = Path(CFG["project"]["data_root"]) / "season" / series
    def _try(p):
        return json.loads(p.read_text()) if p.exists() else {}
    return {
        "M3": _try(season_dir / "M3.json"),
        "M5": _try(season_dir / "M5_personas.json"),
        "M6": _try(season_dir / "M6_relations.json"),
        "M7": _try(season_dir / "M7_world.json"),
    }


def run(series: str, episodes: list[str], tasks: list[str], max_per_task: int = 8) -> None:
    season_ctx = _load_season_ctx(series)
    out_dir = ensure_dir(Path(CFG["project"]["data_root"]) / "qa" / series)
    all_qas: list[dict] = []
    for ep in episodes:
        ep_ctx = _load_episode_ctx(series, ep)
        ctx = {**ep_ctx, **season_ctx}
        for t in tasks:
            seeds_fn = SEED_FNS.get(t)
            if not seeds_fn:
                LOG.warning(f"No seed function for task {t}")
                continue
            seeds = seeds_fn(ep_ctx["events"], ctx)[:max_per_task]
            LOG.info(f"{ep} {t}: {len(seeds)} seeds")
            for s in seeds:
                qa = generate_qa_for_seed(t, s, ep)
                if qa is not None:
                    qa["episode"] = ep
                    qa["series"] = series
                    all_qas.append(qa)
    out_path = out_dir / "qa_staging.jsonl"
    write_jsonl(out_path, all_qas)
    LOG.info(f"Stage 5 done: {len(all_qas)} QAs → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=["T1", "T2", "T4", "T5", "T6", "T7", "T8", "T9", "T10"])
    ap.add_argument("--max_per_task", type=int, default=8)
    args = ap.parse_args()
    run(args.series, args.episodes, args.tasks, args.max_per_task)


if __name__ == "__main__":
    main()
