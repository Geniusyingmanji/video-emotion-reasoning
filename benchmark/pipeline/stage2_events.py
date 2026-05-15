"""Stage 2: Event extraction.

Reads the Stage 1 perception JSONL for an episode and produces a list of events
(affect/action/social) anchored to time spans with multimodal cues.

Strategy:
  Pass A: GPT-5.5 reads the full perception trace (long context) and emits events.
  Pass B: Qwen2.5-Omni Thinker re-reads the audio segments and proposes affect_events
          based on paralinguistic signal (which only the AV model can see).
  Merge:  IoU-based de-dup + semantic merge using a small LLM judge.

Run:
  python stage2_events.py --series breaking_bad --episode tos_dev_4min
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.pipeline.common import (  # noqa: E402
    CFG,
    ensure_dir,
    get_logger,
    llm_chat,
    parse_json_block,
    read_jsonl,
    write_json,
    write_jsonl,
)

LOG = get_logger("stage2")
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "event_extraction.md"


# -----------------------------------------------------------------------------
# Pass A: GPT-5.5 over the entire perception trace
# -----------------------------------------------------------------------------
def _perception_for_llm(perception_rows: list[dict], duration_sec: float) -> str:
    """Render the perception trace as compact text for the LLM (token-friendly)."""
    lines: list[str] = [f"# Episode duration: {duration_sec:.1f}s, {len(perception_rows)} shots"]
    for row in perception_rows:
        ts = f"{row['time_span'][0]:.1f}-{row['time_span'][1]:.1f}s"
        chars = ",".join(row.get("characters_present", [])) or "?"
        lines.append(f"\n## [{row['shot_id']}] {ts}  characters={chars}")
        for u in row.get("utterances", []):
            spk = u.get("speaker_id") or "?"
            tags = " ".join(u.get("paralinguistic", []))
            lines.append(
                f"  [{u['start_sec']:.1f}-{u['end_sec']:.1f}] {spk}: {u['text']}  {tags}".rstrip()
            )
    return "\n".join(lines)


def extract_events_passA(perception_rows: list[dict], meta: dict, episode: str) -> list[dict]:
    """Pass A: GPT-5.5 long-context event extraction."""
    sys_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    user_prompt = (
        f"Episode: {episode}\nDuration: {meta.get('duration_sec', 0):.1f}s\n\n"
        "Below is the perception trace. Return strict JSON array of events.\n\n"
        + _perception_for_llm(perception_rows, meta.get("duration_sec", 0))
    )
    raw = llm_chat(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=8000,
    )
    try:
        events = parse_json_block(raw)
        if isinstance(events, dict) and "events" in events:
            events = events["events"]
        if not isinstance(events, list):
            raise ValueError("Top-level JSON is not a list")
    except Exception as e:  # noqa: BLE001
        LOG.warning(f"Pass-A JSON parse failed: {e}; raw[:300]={raw[:300]!r}")
        events = []
    # Re-id and tag pass
    out: list[dict] = []
    for i, ev in enumerate(events):
        ev["event_id"] = f"{episode}_ev{i:03d}A"
        ev["source"] = "passA_gpt55"
        out.append(ev)
    LOG.info(f"Pass A: {len(out)} events from GPT-5.5")
    return out


# -----------------------------------------------------------------------------
# Pass B: Qwen2.5-Omni over audio segments → affect_events
# -----------------------------------------------------------------------------
def _audio_segments(perception_rows: list[dict], window_sec: float = 60.0) -> list[tuple[float, float]]:
    """Group shots into ~window_sec audio chunks for Pass B."""
    if not perception_rows:
        return []
    total_end = perception_rows[-1]["time_span"][1]
    starts = list(range(0, int(total_end), int(window_sec)))
    return [(s, min(s + window_sec, total_end)) for s in starts]


def extract_events_passB(audio_path: Path, perception_rows: list[dict], episode: str) -> list[dict]:
    """Pass B: Qwen-Omni produces affect_event candidates per audio chunk.

    For each window, ask Qwen-Omni to listen and emit affect events with intensity
    and supporting paralinguistic cues. Visual cues are filled in later from
    Pass A's visual descriptions.
    """
    if not audio_path.exists():
        LOG.warning(f"Audio file missing for Pass B: {audio_path}")
        return []
    import librosa
    import torch
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    mp = CFG["mllm"]["qwen_omni"]["path"]
    processor = Qwen2_5OmniProcessor.from_pretrained(mp)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        mp, torch_dtype=torch.bfloat16, device_map=CFG["mllm"]["qwen_omni"]["device"]
    )
    model.eval()

    audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    sr = 16000

    sys_prompt = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )
    out: list[dict] = []
    windows = _audio_segments(perception_rows, window_sec=60.0)
    for wi, (ws, we) in enumerate(windows):
        chunk = audio[int(ws * sr) : int(we * sr)]
        if len(chunk) < sr:
            continue
        # Characters in this window (from perception)
        chars_in_window = sorted({
            c for row in perception_rows
            if not (row["time_span"][1] < ws or row["time_span"][0] > we)
            for c in row.get("characters_present", [])
        })
        prompt = (
            "Listen for emotional events in this audio window. For each clearly-audible "
            "emotional moment, return JSON object with fields:\n"
            "  start: float seconds (relative to this window start)\n"
            "  end:   float seconds (relative to this window end)\n"
            "  speaker_role: short label (or 'unknown' if unclear)\n"
            "  emotion: one of [anger, fear, sadness, joy, surprise, disgust, guilt, shame, pride, contempt]\n"
            "  intensity: 0.0-1.0\n"
            "  paralinguistic_cues: list of tags from "
            "[tone:trembling, tone:flat, tone:agitated, tone:sarcastic, tone:whispered, "
            " tempo:rushed, tempo:slow, pause:long, nonverbal:sigh, nonverbal:laugh, "
            " nonverbal:cry, nonverbal:cough, volume:loud, volume:soft]\n"
            "  utterance_text: the spoken words (if any)\n"
            f"Characters known to be present in this window: {chars_in_window}\n"
            "Return strict JSON array. Skip moments with no clear emotion. No prose."
        )
        conv = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": [
                {"type": "audio", "audio": chunk},
                {"type": "text", "text": prompt},
            ]},
        ]
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True,
        ).to(model.device)
        with torch.no_grad():
            o = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        gen = processor.batch_decode(o[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        try:
            rows = parse_json_block(gen)
            if not isinstance(rows, list):
                rows = []
        except Exception as e:  # noqa: BLE001
            LOG.warning(f"Pass-B window {wi} parse failed: {e}; raw={gen[:200]!r}")
            rows = []
        for j, r in enumerate(rows):
            try:
                evt = {
                    "event_id": f"{episode}_ev{wi:02d}{j:02d}B",
                    "type": "affect_event",
                    "time_span": [float(r["start"]) + ws, float(r["end"]) + ws],
                    "participants": [r.get("speaker_role", "unknown")],
                    "emotion": r.get("emotion", "neutral"),
                    "intensity": float(r.get("intensity", 0.5)),
                    "trigger_ref": None,
                    "cues": [
                        {"modality": "audio", "ts": [float(r["start"]) + ws, float(r["end"]) + ws],
                         "desc": ", ".join(r.get("paralinguistic_cues", []))}
                    ],
                    "summary": (r.get("utterance_text") or "(non-verbal vocalization)")[:200],
                    "source": "passB_qwen_omni",
                }
                out.append(evt)
            except Exception as e:  # noqa: BLE001
                LOG.warning(f"row {j} skipped: {e}; row={r}")
    LOG.info(f"Pass B: {len(out)} affect_event candidates from Qwen-Omni")
    return out


# -----------------------------------------------------------------------------
# Merge passes via IoU + semantic LLM judge
# -----------------------------------------------------------------------------
def _iou(a: list[float], b: list[float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1e-6
    return inter / union


def merge_events(events_a: list[dict], events_b: list[dict]) -> list[dict]:
    """Merge passes A/B. Keep all of A; for each B, attach to nearest A if IoU > thr; otherwise add as new."""
    thr = CFG["events"]["iou_merge_threshold"]
    merged: list[dict] = list(events_a)
    used_b: set[int] = set()
    for bi, b in enumerate(events_b):
        if b.get("type") != "affect_event":
            continue
        best, best_iou = None, 0.0
        for ai, a in enumerate(merged):
            if a.get("type") != "affect_event":
                continue
            iou = _iou(a["time_span"], b["time_span"])
            if iou > best_iou:
                best, best_iou = ai, iou
        if best is not None and best_iou >= thr:
            # Augment A with B's audio cue and emotion confirmation
            a = merged[best]
            a.setdefault("cues", []).extend(b.get("cues", []))
            a.setdefault("_audio_confirmation", []).append({
                "emotion": b["emotion"],
                "intensity": b["intensity"],
                "from_event_id": b["event_id"],
            })
            used_b.add(bi)
        else:
            merged.append(b)
    LOG.info(f"Merged: kept {len(merged)} events total ({len(events_a)} A + {len(events_b) - len(used_b)} new B)")
    return merged


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def run(series: str, episode: str, skip_passB: bool = False) -> None:
    perception_path = Path(CFG["project"]["data_root"]) / "perception" / series / f"{episode}.jsonl"
    meta_path = Path(CFG["project"]["data_root"]) / "perception" / series / f"{episode}.meta.json"
    audio_path = Path(CFG["project"]["data_root"]) / "perception" / series / "audio" / f"{episode}.wav"
    if not perception_path.exists():
        raise FileNotFoundError(f"Run Stage 1 first: missing {perception_path}")
    rows = read_jsonl(perception_path)
    meta = json.loads(meta_path.read_text())
    LOG.info(f"Loaded perception: {len(rows)} shots, {meta.get('duration_sec', 0):.1f}s")

    events_a = extract_events_passA(rows, meta, episode)
    events_b = [] if skip_passB else extract_events_passB(audio_path, rows, episode)
    merged = merge_events(events_a, events_b)

    out_dir = ensure_dir(Path(CFG["project"]["data_root"]) / "events" / series)
    write_jsonl(out_dir / f"{episode}.jsonl", merged)
    write_json(out_dir / f"{episode}.meta.json", {
        "series": series,
        "episode": episode,
        "n_passA": len(events_a),
        "n_passB": len(events_b),
        "n_merged": len(merged),
        "n_affect_events": sum(1 for e in merged if e.get("type") == "affect_event"),
        "n_action_events": sum(1 for e in merged if e.get("type") == "action_event"),
        "n_social_events": sum(1 for e in merged if e.get("type") == "social_event"),
    })
    LOG.info(f"Stage 2 done: {out_dir / f'{episode}.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--skip_passB", action="store_true",
                    help="Skip Qwen-Omni audio pass (faster dev iteration)")
    args = ap.parse_args()
    run(args.series, args.episode, args.skip_passB)


if __name__ == "__main__":
    main()
