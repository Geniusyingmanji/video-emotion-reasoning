"""Stage 7: Evaluation panel — evaluate baseline models across 4 settings (E0/E1/E2/E3).

For each QA in qa_filtered.jsonl, build the appropriate context for each setting
and ask the model to answer A/B/C/D. Record predictions and compute per-task,
per-setting accuracy.

Settings:
  E0: only the local event (±2-5min clip context as captions/utterances)
  E1: full episode (all perception + events for the episode)
  E2: full season (E1 + cross-episode summaries from M3 + M4 cumulative)
  E3: full season + explicit M5/M6/M7 semantic memory injected

Models are configured in benchmark/configs/config.yaml under `eval.models`.
Default uses just the LiteLLM proxy GPT-5.5 for now; users add more via config.

Run:
  python stage7_eval.py --series synthetic_demo --setting E0
  python stage7_eval.py --series synthetic_demo --setting E1
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.pipeline.common import (  # noqa: E402
    CFG,
    ensure_dir,
    get_logger,
    llm_chat,
    read_jsonl,
    write_json,
    write_jsonl,
)

LOG = get_logger("stage7")


# -----------------------------------------------------------------------------
# Context builders per setting
# -----------------------------------------------------------------------------
def _load_episode_data(series: str, episode: str) -> dict:
    data_root = Path(CFG["project"]["data_root"])
    perception_path = data_root / "perception" / series / f"{episode}.jsonl"
    events_path = data_root / "events" / series / f"{episode}.jsonl"
    return {
        "perception": read_jsonl(perception_path) if perception_path.exists() else [],
        "events": read_jsonl(events_path) if events_path.exists() else [],
    }


def _load_season_data(series: str) -> dict:
    season_dir = Path(CFG["project"]["data_root"]) / "season" / series
    def _try(p):
        return json.loads(p.read_text()) if p.exists() else {}
    return {
        "M3": _try(season_dir / "M3.json"),
        "M4_index": _try(season_dir / "M4_index.json"),
        "M5": _try(season_dir / "M5_personas.json"),
        "M6": _try(season_dir / "M6_relations.json"),
        "M7": _try(season_dir / "M7_world.json"),
    }


def build_context_E0(qa: dict, ep_data: dict) -> str:
    """E0: ±2min around the answer_evidence event."""
    ae = qa.get("answer_evidence", {})
    ev_id = ae.get("event_id") or ae.get("seed_event_id")
    target_event = next((e for e in ep_data["events"] if e["event_id"] == ev_id), None)
    if target_event is None:
        # Fallback: use first time_span we can find
        return "Context unavailable (E0)."
    t0 = max(0, target_event["time_span"][0] - 120)
    t1 = target_event["time_span"][1] + 120

    lines: list[str] = [f"Local context (±2 min around event):"]
    for row in ep_data["perception"]:
        if row["time_span"][1] < t0 or row["time_span"][0] > t1:
            continue
        lines.append(f"  Shot {row['shot_id']} [{row['time_span'][0]:.1f}-{row['time_span'][1]:.1f}] chars={row['characters_present']}")
        for u in row["utterances"]:
            tags = " ".join(u.get("paralinguistic", []))
            lines.append(f"    {u['start_sec']:.1f}-{u['end_sec']:.1f} {u.get('speaker_id') or '?'}: {u['text']}  {tags}")
    return "\n".join(lines)


def build_context_E1(qa: dict, ep_data: dict) -> str:
    """E1: full episode perception + events."""
    lines: list[str] = [
        f"Full episode context (all {len(ep_data['perception'])} shots, {len(ep_data['events'])} events):",
        "## Events:",
    ]
    for e in ep_data["events"]:
        lines.append(
            f"  {e['event_id']} [{e['time_span'][0]:.1f}-{e['time_span'][1]:.1f}] "
            f"{e['type']} participants={e.get('participants', [])} "
            f"emotion={e.get('emotion')} | {e.get('summary', '')[:200]}"
        )
    lines.append("\n## Dialogue / paralanguage:")
    for row in ep_data["perception"]:
        for u in row["utterances"]:
            tags = " ".join(u.get("paralinguistic", []))
            lines.append(f"  [{u['start_sec']:.1f}-{u['end_sec']:.1f}] {u.get('speaker_id') or '?'}: {u['text']}  {tags}")
    return "\n".join(lines)


def build_context_E2(qa: dict, ep_data: dict, season_data: dict) -> str:
    """E2: E1 + cross-episode summary from M3 + M4 cumulative trajectories (summary form)."""
    parts: list[str] = [build_context_E1(qa, ep_data)]
    M3 = season_data.get("M3", {})
    if M3.get("cross_episode_edges"):
        parts.append("\n## Cross-episode edges (M3, summary):")
        for ed in M3["cross_episode_edges"][:80]:
            parts.append(f"  {ed.get('src_event_id')} → {ed.get('dst_event_id')} [{ed.get('type')}]")
    return "\n".join(parts)


def build_context_E3(qa: dict, ep_data: dict, season_data: dict) -> str:
    """E3: E2 + explicit M5 (OCEAN personas) + M6 (relations) + M7 (world facts)."""
    parts: list[str] = [build_context_E2(qa, ep_data, season_data)]

    M5 = season_data.get("M5", {})
    if M5:
        parts.append("\n## M5 — character personas (OCEAN):")
        for char, p in M5.items():
            if not isinstance(p, dict) or "ocean" not in p:
                continue
            ocean_summary = ", ".join(
                f"{k[:3].upper()}={v.get('score')}" for k, v in p["ocean"].items()
            )
            parts.append(f"  {char}: {ocean_summary}")
            parts.append(f"    narrative: {p.get('narrative_summary', '')[:240]}")

    M6 = season_data.get("M6", {})
    if M6.get("edges"):
        parts.append("\n## M6 — relationship network:")
        for e in M6["edges"]:
            parts.append(
                f"  {e.get('src')} → {e.get('dst')}: {e.get('relation_type')} "
                f"(intensity={e.get('intensity')})"
            )

    M7 = season_data.get("M7", {})
    if M7.get("facts"):
        parts.append("\n## M7 — propositional world/plot state:")
        for f in M7["facts"][:80]:
            parts.append(f"  - [{f.get('category', '?')}] {f.get('statement', '')}")

    return "\n".join(parts)


CONTEXT_BUILDERS = {
    "E0": build_context_E0,
    "E1": build_context_E1,
    "E2": build_context_E2,
    "E3": build_context_E3,
}


# -----------------------------------------------------------------------------
# Single-QA evaluation
# -----------------------------------------------------------------------------
def evaluate_one(qa: dict, setting: str, ep_data: dict, season_data: dict, model: str) -> dict:
    if setting in {"E0", "E1"}:
        ctx = CONTEXT_BUILDERS[setting](qa, ep_data)
    else:
        ctx = CONTEXT_BUILDERS[setting](qa, ep_data, season_data)

    sys_prompt = (
        "You are a careful TV-show analyst answering an MCQ. Choose the best option from A/B/C/D. "
        "Output ONLY the single letter A B C or D, nothing else."
    )
    user = (
        f"Setting: {setting}\n\n"
        f"{ctx}\n\n---\n"
        f"Q: {qa['question']}\n\n"
        f"Options:\n"
        + "\n".join(f"  {k}: {v}" for k, v in qa['options'].items())
    )
    try:
        raw = llm_chat(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user},
            ],
            model=model,
            max_tokens=64,  # GPT-5.5 uses reasoning tokens; <20 returns empty content
            temperature=0.0,
        )
        # The model often outputs whitespace + letter (or reasoning + letter); pick first A/B/C/D
        cand = (raw or "").upper()
        pred = None
        for ch in cand:
            if ch in "ABCD":
                pred = ch
                break
    except Exception as e:  # noqa: BLE001
        LOG.warning(f"Eval failed on {qa['qid']}: {e}")
        pred = None
    return {
        "qid": qa.get("qid"),
        "task": qa.get("task"),
        "series": qa.get("series"),
        "episode": qa.get("episode"),
        "setting": setting,
        "model": model,
        "predicted": pred,
        "correct_answer": qa.get("correct"),
        "is_correct": pred == qa.get("correct"),
        "memory_components_claimed": qa.get("answer_evidence", {}).get("memory_components"),
    }


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def run(series: str, episode: str, setting: str, in_qa: Path, model: str | None = None) -> None:
    model = model or CFG["llm"]["primary_model"]
    LOG.info(f"Stage 7 eval: {series}/{episode} setting={setting} model={model}")
    qas = read_jsonl(in_qa)
    ep_data = _load_episode_data(series, episode)
    season_data = _load_season_data(series) if setting in {"E2", "E3"} else {}

    results: list[dict] = []
    for qa in qas:
        if qa.get("episode") and qa["episode"] != episode:
            continue
        res = evaluate_one(qa, setting, ep_data, season_data, model)
        results.append(res)

    # Aggregate
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_task[r["task"]].append(r)
    per_task_acc = {
        t: sum(r["is_correct"] for r in rs if r["predicted"] is not None) / max(1, len(rs))
        for t, rs in by_task.items()
    }
    overall = sum(r["is_correct"] for r in results if r["predicted"] is not None) / max(1, len(results))

    out_dir = ensure_dir(Path(CFG["project"]["data_root"]) / "eval" / series)
    out_path = out_dir / f"{episode}_{setting}_{model.replace('/', '_')}.jsonl"
    write_jsonl(out_path, results)
    summary_path = out_dir / f"{episode}_{setting}_{model.replace('/', '_')}.summary.json"
    write_json(summary_path, {
        "series": series, "episode": episode, "setting": setting, "model": model,
        "n_questions": len(results),
        "overall_accuracy": overall,
        "per_task_accuracy": per_task_acc,
    })
    LOG.info(f"Stage 7 done: overall={overall:.3f}, per_task={per_task_acc} → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--episode", required=True, help="restrict eval to QAs for this episode")
    ap.add_argument("--setting", required=True, choices=["E0", "E1", "E2", "E3"])
    ap.add_argument("--in_qa", type=Path,
                    default=None,
                    help="default: data/final/<series>/qa_filtered.jsonl")
    ap.add_argument("--model", default=None,
                    help="LiteLLM model name; default from config primary_model")
    args = ap.parse_args()
    in_qa = args.in_qa or (
        Path(CFG["project"]["data_root"]) / "final" / args.series / "qa_filtered.jsonl"
    )
    run(args.series, args.episode, args.setting, in_qa, args.model)


if __name__ == "__main__":
    main()
