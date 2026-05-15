"""Stage 3: Per-episode event relation extraction → M2 event DAG.

Dual-pass relation candidates (we re-use GPT-5.5 for now; Qwen-Omni vote
is optional for affect→action edges where audio matters).
Then intersect votes to high-confidence edges + DAG-enforce + temporal-monotonic
+ confidence threshold.

Run:
  python stage3_relations.py --series breaking_bad --episode tos_dev_4min
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

LOG = get_logger("stage3")
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "relation_extraction.md"


def _events_for_llm(events: list[dict]) -> str:
    """Compact event listing for the LLM."""
    lines: list[str] = []
    for e in events:
        lines.append(
            f"- {e['event_id']} [{e['time_span'][0]:.1f}-{e['time_span'][1]:.1f}] "
            f"{e['type']} participants={e.get('participants', [])} "
            f"emotion={e.get('emotion')} intensity={e.get('intensity')} | {e.get('summary', '')[:150]}"
        )
    return "\n".join(lines)


def extract_edges(events: list[dict], variant_name: str) -> list[dict]:
    """One pass of edge extraction."""
    sys_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    user_prompt = (
        f"Variant: {variant_name}\n\n"
        "Events:\n" + _events_for_llm(events) + "\n\n"
        "Return JSON: {\"edges\": [{src, dst, type, confidence, evidence}, ...]}"
    )
    raw = llm_chat(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=6000,
    )
    try:
        obj = parse_json_block(raw)
        edges = obj.get("edges", obj if isinstance(obj, list) else [])
    except Exception as e:  # noqa: BLE001
        LOG.warning(f"Edge parse ({variant_name}) failed: {e}; raw[:200]={raw[:200]!r}")
        edges = []
    LOG.info(f"{variant_name}: {len(edges)} edges")
    return edges


def _edge_key(e: dict) -> tuple[str, str, str]:
    return (e["src"], e["dst"], e["type"])


def consolidate_edges(edges_pass1: list[dict], edges_pass2: list[dict], events: list[dict]) -> dict:
    """Intersect = high-confidence; union = candidate.

    Then enforce:
      - DAG: drop edges that create cycles (in order of confidence ascending).
      - Temporal monotonicity: src.time_span[0] <= dst.time_span[0]; drop violators.
    """
    id_to_event = {e["event_id"]: e for e in events}
    set1 = {_edge_key(e): e for e in edges_pass1}
    set2 = {_edge_key(e): e for e in edges_pass2}
    intersect_keys = set(set1) & set(set2)
    union_keys = set(set1) | set(set2)

    def merge_pair(e1: dict, e2: dict) -> dict:
        return {
            "src": e1["src"], "dst": e1["dst"], "type": e1["type"],
            "confidence": (float(e1.get("confidence", 0.5)) + float(e2.get("confidence", 0.5))) / 2,
            "evidence": e1.get("evidence", "") + " || " + e2.get("evidence", ""),
            "votes": 2,
        }

    high_conf = [merge_pair(set1[k], set2[k]) for k in intersect_keys]
    cand_keys = union_keys - intersect_keys
    candidates = []
    for k in cand_keys:
        e = set1.get(k) or set2[k]
        candidates.append({**e, "votes": 1})

    # Temporal monotonicity
    def temporally_valid(e: dict) -> bool:
        s, d = id_to_event.get(e["src"]), id_to_event.get(e["dst"])
        if not s or not d:
            return False
        return s["time_span"][0] <= d["time_span"][0]

    high_conf = [e for e in high_conf if temporally_valid(e)]
    candidates = [e for e in candidates if temporally_valid(e)]

    # DAG check via Kahn's algorithm; drop low-confidence edges that create cycles
    def cycle_drop(edges: list[dict]) -> list[dict]:
        from collections import defaultdict, deque

        edges_sorted = sorted(edges, key=lambda e: -float(e.get("confidence", 0.5)))
        kept: list[dict] = []
        adj: dict[str, set[str]] = defaultdict(set)
        for e in edges_sorted:
            # Tentatively add
            adj[e["src"]].add(e["dst"])
            # Detect cycle: BFS from dst to see if we can reach src
            visited: set[str] = set()
            q = deque([e["dst"]])
            cycle = False
            while q:
                u = q.popleft()
                if u == e["src"]:
                    cycle = True
                    break
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
            if cycle:
                adj[e["src"]].discard(e["dst"])
            else:
                kept.append(e)
        return kept

    high_conf = cycle_drop(high_conf)
    candidates = cycle_drop(candidates + high_conf)  # candidates considered alongside

    return {
        "high_confidence_edges": high_conf,
        "candidate_edges": [e for e in candidates if e not in high_conf],
        "n_intersect": len(intersect_keys),
        "n_union": len(union_keys),
        "n_high_conf_kept": len(high_conf),
    }


def run(series: str, episode: str) -> None:
    events_path = Path(CFG["project"]["data_root"]) / "events" / series / f"{episode}.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"Run Stage 2 first: missing {events_path}")
    events = read_jsonl(events_path)
    LOG.info(f"Loaded {len(events)} events for {episode}")

    # Two LLM passes with slightly different framing to surface disagreement.
    edges_p1 = extract_edges(events, variant_name="primary")
    edges_p2 = extract_edges(events, variant_name="adversarial")  # we just re-run; randomness via temperature

    result = consolidate_edges(edges_p1, edges_p2, events)
    out_dir = ensure_dir(Path(CFG["project"]["data_root"]) / "event_graph" / series)
    M2 = {
        "series": series,
        "episode": episode,
        "events": [e["event_id"] for e in events],
        **result,
    }
    write_json(out_dir / f"{episode}.M2.json", M2)
    LOG.info(
        f"Stage 3 done: high_conf={len(result['high_confidence_edges'])} "
        f"candidate={len(result['candidate_edges'])} → {out_dir / f'{episode}.M2.json'}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--episode", required=True)
    args = ap.parse_args()
    run(args.series, args.episode)


if __name__ == "__main__":
    main()
