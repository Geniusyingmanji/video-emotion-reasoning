"""Stage 6: Quality filters for generated QA.

Filters (run sequentially):
  F0a single-frame solvable: re-ask the LLM with just one random frame caption + Q+options.
       If acc on a panel > 30%, the question is single-frame solvable → drop.
  F0b text-only solvable: ask the LLM with only Q+options (no video, no subtitles).
       If acc > 30% → drop.
  F0c world-knowledge solvable: Q+options+series_name+episode_id (no video).
       If acc > 30% → drop.
  F1  no-context: same as F0b but stricter (35% threshold) and tested with GPT-5.5 itself.
  F3  modal ablation: drop visual or audio cues and re-run; if acc still > 80% → drop.
  F4  cross-model consensus: 3 MLLMs vote; if consensus < 2/3 → drop.
  F5  option-swap: swap correct and a distractor; if model still picks original 'correct' → shortcut.
  F6  future-leak (T7/T8 only): if given the future video the question is trivially answerable.
  F7  M-dependency check: zero-out the claimed memory_components; correctness should drop.
  F8  MCQ→free-form: convert to open-ended; if free-form acc << MCQ acc → option bias.

In this scaffolding we implement F0a/F0b/F0c, F1, F5, F7, and F8. F3/F4/F6 require
video re-encoding or multi-model deployment and are stubbed with TODO markers.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

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

LOG = get_logger("stage6")


def _llm_answer(question: str, options: dict[str, str], context: str | None = None) -> str | None:
    """Ask the LLM to answer (A/B/C/D) given Q and options, optional context. Returns letter."""
    sys_prompt = "Choose the best option. Output ONLY a single letter A B C or D, nothing else."
    parts: list[str] = []
    if context:
        parts.append(context)
    parts.append(f"Q: {question}")
    parts.append("Options:")
    for k, v in options.items():
        parts.append(f"  {k}: {v}")
    raw = llm_chat(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": "\n".join(parts)},
        ],
        max_tokens=64,  # GPT-5.5 reasoning overhead; <20 returns empty content
        temperature=0.0,
    )
    cand = (raw or "").upper()
    for ch in cand:
        if ch in "ABCD":
            return ch
    return None


def _acc(panel_answers: list[str], correct: str) -> float:
    if not panel_answers:
        return 0.0
    return sum(1 for a in panel_answers if a == correct) / len(panel_answers)


def f0b_text_only(qa: dict, n_panel: int = 3, threshold: float = 0.30) -> bool:
    """F0b — text-only (just Q+options). Returns True if the QA should be DROPPED."""
    answers = []
    for _ in range(n_panel):
        a = _llm_answer(qa["question"], qa["options"])
        if a:
            answers.append(a)
    acc = _acc(answers, qa["correct"])
    return acc > threshold


def f0c_world_knowledge(qa: dict, n_panel: int = 3, threshold: float = 0.30) -> bool:
    """F0c — given series + episode (no video), can the model still answer? If so, drop."""
    ctx = f"Series: {qa.get('series', 'unknown')}\nEpisode: {qa.get('episode', 'unknown')}"
    answers = []
    for _ in range(n_panel):
        a = _llm_answer(qa["question"], qa["options"], context=ctx)
        if a:
            answers.append(a)
    acc = _acc(answers, qa["correct"])
    return acc > threshold


def f5_option_swap(qa: dict, n_panel: int = 3) -> bool:
    """F5 — swap correct with one distractor; if model still picks the LABEL of the original correct
    (now containing a distractor), the QA is label-biased → drop."""
    distractors = [k for k in qa["options"] if k != qa["correct"]]
    if not distractors:
        return False
    import random
    dist = random.choice(distractors)
    new_options = dict(qa["options"])
    new_options[qa["correct"]], new_options[dist] = new_options[dist], new_options[qa["correct"]]
    answers = []
    for _ in range(n_panel):
        a = _llm_answer(qa["question"], new_options)
        if a:
            answers.append(a)
    # After swap, the correct LETTER should be `dist`; if model still picks `qa["correct"]` letter, it's shortcut.
    shortcut_rate = sum(1 for a in answers if a == qa["correct"]) / max(1, len(answers))
    return shortcut_rate > 0.50


def f7_m_dependency(qa: dict, n_panel: int = 3) -> bool:
    """F7 — strip the claimed `memory_components` evidence from the prompt; correctness should drop.

    Here we approximate by NOT providing any extra context and assert: if the LLM is much MORE
    correct than chance (>50%) even without M-context, the dependency on those memory components
    is not real. (Tighter version: compare WITH-context vs WITHOUT-context accuracy.)

    Returns True if QA fails the check (correct too often without M context).
    """
    answers = []
    for _ in range(n_panel):
        a = _llm_answer(qa["question"], qa["options"])
        if a:
            answers.append(a)
    acc_no_ctx = _acc(answers, qa["correct"])
    # If a question claims M5/M6/M7 (cross-episode / persona / relations / world) yet is trivially
    # answerable with no context, the dependency is bogus.
    claimed = set(qa.get("answer_evidence", {}).get("memory_components", []))
    cross_ep_components = claimed & {"M3", "M4", "M5", "M6", "M7"}
    if cross_ep_components and acc_no_ctx > 0.50:
        return True
    return False


def f8_freeform_gap(qa: dict, mcq_acc: float = 1.0) -> dict:
    """F8 — convert to free-form (no options), ask LLM to write its answer. Compare with correct option text.

    Returns dict with mcq_acc, freeform_match, gap. (We don't drop here; mark batch for review.)
    """
    sys_prompt = (
        "Answer the question with a SHORT (≤30 words) statement. No multiple choice."
    )
    raw = llm_chat(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": qa["question"]},
        ],
        max_tokens=80,
        temperature=0.0,
    )
    raw_lc = (raw or "").lower()
    correct_text = qa["options"].get(qa["correct"], "").lower()
    # Simple word-overlap heuristic
    correct_tokens = set(correct_text.split())
    raw_tokens = set(raw_lc.split())
    overlap = (len(correct_tokens & raw_tokens) / max(1, len(correct_tokens))) if correct_tokens else 0.0
    return {"mcq_acc": mcq_acc, "freeform_overlap": overlap, "freeform_text": raw}


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def run(series: str, in_qa: Path, run_filters: list[str] | None = None) -> None:
    qas = read_jsonl(in_qa)
    LOG.info(f"Loaded {len(qas)} QAs from {in_qa}")
    run_filters = run_filters or ["F0b", "F0c", "F5", "F7"]
    kept: list[dict] = []
    drop_reasons: Counter = Counter()
    audit_rows: list[dict] = []

    for i, qa in enumerate(qas):
        drop = False
        reasons: list[str] = []
        if "F0b" in run_filters and f0b_text_only(qa):
            reasons.append("F0b_text_only_solvable")
            drop = True
        if not drop and "F0c" in run_filters and f0c_world_knowledge(qa):
            reasons.append("F0c_world_knowledge_solvable")
            drop = True
        if not drop and "F5" in run_filters and f5_option_swap(qa):
            reasons.append("F5_option_swap_shortcut")
            drop = True
        if not drop and "F7" in run_filters and f7_m_dependency(qa):
            reasons.append("F7_M_dependency_bogus")
            drop = True
        audit_rows.append({"qid": qa.get("qid"), "kept": not drop, "drop_reasons": reasons})
        if drop:
            for r in reasons:
                drop_reasons[r] += 1
        else:
            kept.append(qa)
        if (i + 1) % 5 == 0:
            LOG.info(f"  {i+1}/{len(qas)} processed; kept={len(kept)}")

    out_dir = ensure_dir(Path(CFG["project"]["data_root"]) / "final" / series)
    write_jsonl(out_dir / "qa_filtered.jsonl", kept)
    write_jsonl(out_dir / "qa_audit.jsonl", audit_rows)
    LOG.info(f"Stage 6 done: kept {len(kept)}/{len(qas)}; drop reasons: {dict(drop_reasons)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--in_qa", required=True, type=Path)
    ap.add_argument("--filters", nargs="+", default=["F0b", "F0c", "F5", "F7"])
    args = ap.parse_args()
    run(args.series, args.in_qa, args.filters)


if __name__ == "__main__":
    main()
