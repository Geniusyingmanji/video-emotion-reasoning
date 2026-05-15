"""Stage 7 evaluator using local Qwen2.5-Omni-7B Thinker.

Unlike stage7_eval.py which calls the LiteLLM proxy (GPT-5.5, text-only context),
this variant uses Qwen2.5-Omni Thinker for native audio+text reasoning on E0
(local clip). For E1/E2/E3 the audio is too long to fit (Qwen-Omni has ~30s
practical audio limit per call), so we fall back to text context.

Run:
  python -m benchmark.pipeline.stage7_eval_qwen_omni \\
      --series synthetic_demo --episode ep01_demo --setting E0
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Disable cuDNN BEFORE importing torch-using libraries (see Stage 1 note).
try:
    import torch as _t
    _t.backends.cudnn.enabled = False
except Exception:  # noqa: BLE001
    pass

from benchmark.pipeline.common import (  # noqa: E402
    CFG,
    ensure_dir,
    get_logger,
    parse_json_block,
    read_jsonl,
    write_json,
    write_jsonl,
)
from benchmark.pipeline.stage7_eval import (  # noqa: E402
    _load_episode_data,
    _load_season_data,
    build_context_E1,
    build_context_E2,
    build_context_E3,
)

LOG = get_logger("stage7_qwen_omni")
MODEL_NAME = "qwen2.5-omni-7b-thinker"


_QWEN = None  # cached (model, processor)


def _load() -> tuple:
    global _QWEN
    if _QWEN is not None:
        return _QWEN
    import torch
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    mp = CFG["mllm"]["qwen_omni"]["path"]
    LOG.info(f"Loading Qwen2.5-Omni Thinker from {mp}")
    proc = Qwen2_5OmniProcessor.from_pretrained(mp)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        mp, torch_dtype=torch.bfloat16, device_map=CFG["mllm"]["qwen_omni"]["device"]
    )
    model.eval()
    _QWEN = (model, proc)
    return _QWEN


def _load_audio_slice(series: str, episode: str, t0: float, t1: float) -> "np.ndarray | None":
    import librosa
    audio_path = Path(CFG["project"]["data_root"]) / "perception" / series / "audio" / f"{episode}.wav"
    if not audio_path.exists():
        return None
    audio, _ = librosa.load(str(audio_path), sr=16000, mono=True, offset=t0, duration=max(0.1, t1 - t0))
    return audio


def evaluate_one(
    qa: dict,
    setting: str,
    ep_data: dict,
    season_data: dict,
    audio: "np.ndarray | None",
    text_context: str,
) -> dict:
    import torch
    from qwen_omni_utils import process_mm_info

    model, processor = _load()

    sys_prompt = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )
    qa_text = (
        f"Setting: {setting}\n\n{text_context}\n\n---\n"
        f"Q: {qa['question']}\n\n"
        f"Options:\n"
        + "\n".join(f"  {k}: {v}" for k, v in qa['options'].items())
        + "\n\nReply with ONLY a single letter A, B, C, or D."
    )

    content: list[dict] = []
    if audio is not None and len(audio) >= 16000 * 0.5:
        content.append({"type": "audio", "audio": audio})
    content.append({"type": "text", "text": qa_text})

    conv = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user", "content": content},
    ]
    try:
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=32, do_sample=False, repetition_penalty=1.2,
            )
        gen = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    except Exception as e:  # noqa: BLE001
        LOG.warning(f"qwen-omni eval failed on {qa['qid']}: {e}")
        gen = ""
    pred = None
    for ch in (gen or "").upper():
        if ch in "ABCD":
            pred = ch
            break
    return {
        "qid": qa.get("qid"),
        "task": qa.get("task"),
        "series": qa.get("series"),
        "episode": qa.get("episode"),
        "setting": setting,
        "model": MODEL_NAME,
        "predicted": pred,
        "correct_answer": qa.get("correct"),
        "is_correct": pred == qa.get("correct"),
        "raw_output": (gen or "")[:200],
        "used_audio": audio is not None,
        "memory_components_claimed": qa.get("answer_evidence", {}).get("memory_components"),
    }


def run(series: str, episode: str, setting: str, in_qa: Path) -> None:
    LOG.info(f"Stage 7 (Qwen-Omni) eval: {series}/{episode} setting={setting}")
    qas = read_jsonl(in_qa)
    ep_data = _load_episode_data(series, episode)
    season_data = _load_season_data(series) if setting in {"E2", "E3"} else {}

    results: list[dict] = []
    for qa in qas:
        if qa.get("episode") and qa["episode"] != episode:
            continue
        # Build context per setting
        if setting == "E0":
            # Use audio window ±60s around the seed event
            ae = qa.get("answer_evidence", {})
            ev_id = ae.get("event_id") or ae.get("seed_event_id")
            target = next((e for e in ep_data["events"] if e["event_id"] == ev_id), None)
            if target:
                t0 = max(0.0, target["time_span"][0] - 30.0)
                t1 = target["time_span"][1] + 30.0
            else:
                t0, t1 = 0.0, 30.0
            audio = _load_audio_slice(series, episode, t0, t1)
            text_ctx = f"Audio window from {t0:.1f}-{t1:.1f}s (provided as audio input above)."
        elif setting == "E1":
            audio = None  # too long for Qwen-Omni
            text_ctx = build_context_E1(qa, ep_data)
        elif setting == "E2":
            audio = None
            text_ctx = build_context_E2(qa, ep_data, season_data)
        else:  # E3
            audio = None
            text_ctx = build_context_E3(qa, ep_data, season_data)
        res = evaluate_one(qa, setting, ep_data, season_data, audio, text_ctx)
        results.append(res)

    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_task[r["task"]].append(r)
    per_task_acc = {
        t: sum(r["is_correct"] for r in rs if r["predicted"] is not None) / max(1, len(rs))
        for t, rs in by_task.items()
    }
    overall = sum(r["is_correct"] for r in results if r["predicted"] is not None) / max(1, len(results))

    out_dir = ensure_dir(Path(CFG["project"]["data_root"]) / "eval" / series)
    out_path = out_dir / f"{episode}_{setting}_{MODEL_NAME}.jsonl"
    write_jsonl(out_path, results)
    write_json(out_dir / f"{episode}_{setting}_{MODEL_NAME}.summary.json", {
        "series": series, "episode": episode, "setting": setting, "model": MODEL_NAME,
        "n_questions": len(results),
        "overall_accuracy": overall,
        "per_task_accuracy": per_task_acc,
    })
    LOG.info(f"Stage 7 Qwen-Omni done: overall={overall:.3f}, per_task={per_task_acc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--setting", required=True, choices=["E0", "E1", "E2", "E3"])
    ap.add_argument("--in_qa", type=Path, default=None)
    args = ap.parse_args()
    in_qa = args.in_qa or (
        Path(CFG["project"]["data_root"]) / "final" / args.series / "qa_filtered.jsonl"
    )
    run(args.series, args.episode, args.setting, in_qa)


if __name__ == "__main__":
    main()
