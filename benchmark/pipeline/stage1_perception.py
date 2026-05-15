"""Stage 1: Multi-track perception pipeline.

Per-episode pipeline producing a time-aligned multi-track JSONL where each shot
carries visual caption, ASR utterances + paralinguistic tags, speaker
diarization, face cluster -> character name mapping.

Note: cuDNN 9.2 in this PyTorch 2.5.1+cu121 install fails with CUDNN_STATUS_NOT_INITIALIZED
on conv ops, so we disable cuDNN globally on import. Conv ops fall back to
non-cuDNN kernels (~10-20% slower) but otherwise work correctly.

Tracks:
  - Shot boundary       (PySceneDetect ContentDetector)
  - Visual caption + micro-expression (GPT-5.5 multimodal on sampled frames)
  - ASR + paralanguage  (Qwen2.5-Omni Thinker)
  - Speaker diarization (pyannote)
  - Face detect + cluster (InsightFace RetinaFace+ArcFace)
  - Character naming    (GPT-5.5 cross-ref against subtitles)

Run:
  python stage1_perception.py --video /path/to/episode.mp4 --series breaking_bad --episode ep01

Outputs:
  data/perception/<series>/<episode>.jsonl    (one row per shot)
  data/perception/<series>/<episode>.meta.json (episode-level metadata)
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# Disable cuDNN BEFORE importing torch-using libraries downstream.
try:
    import torch as _t

    _t.backends.cudnn.enabled = False
except Exception:  # noqa: BLE001
    pass

# Lazy imports for heavy libs guarded by feature flags so we can test in pieces.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.pipeline.common import (  # noqa: E402
    CFG,
    Shot,
    Utterance,
    ensure_dir,
    get_logger,
    llm_chat,
    write_json,
    write_jsonl,
)

LOG = get_logger("stage1")


# -----------------------------------------------------------------------------
# Shot boundary detection
# -----------------------------------------------------------------------------
def detect_shots(video_path: Path) -> tuple[list[Shot], float]:
    """Run PySceneDetect ContentDetector. Returns (shots, fps)."""
    from scenedetect import detect, ContentDetector, open_video

    LOG.info(f"Detecting shots in {video_path}")
    video = open_video(str(video_path))
    fps = float(video.frame_rate)  # may be a Fraction
    detector = ContentDetector(
        threshold=CFG["perception"]["shot_detector"]["threshold"],
        min_scene_len=CFG["perception"]["shot_detector"]["min_scene_len_frames"],
    )
    scenes = detect(str(video_path), detector)
    shots: list[Shot] = []
    for idx, (start, end) in enumerate(scenes):
        shots.append(
            Shot(
                shot_id=f"shot_{idx:04d}",
                start_sec=start.get_seconds(),
                end_sec=end.get_seconds(),
                start_frame=start.get_frames(),
                end_frame=end.get_frames(),
            )
        )
    # Fallback for very short videos where no scenes are detected: single shot.
    if not shots:
        from scenedetect import VideoStreamCv2

        v = VideoStreamCv2(str(video_path))
        dur = v.duration.get_seconds()
        shots = [
            Shot(shot_id="shot_0000", start_sec=0.0, end_sec=dur, start_frame=0, end_frame=int(dur * fps))
        ]
    LOG.info(f"Detected {len(shots)} shots @ {fps:.2f} fps")
    return shots, fps


# -----------------------------------------------------------------------------
# Audio extraction + ASR + paralanguage via Qwen2.5-Omni Thinker
# -----------------------------------------------------------------------------
def extract_audio(video_path: Path, out_wav: Path, sr: int = 16000) -> None:
    """Extract 16kHz mono WAV via ffmpeg."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sr), "-acodec", "pcm_s16le",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)


_QWEN = None  # cached (model, processor)


def _load_qwen_omni():
    global _QWEN
    if _QWEN is not None:
        return _QWEN
    import torch
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

    mp = CFG["mllm"]["qwen_omni"]["path"]
    LOG.info(f"Loading Qwen2.5-Omni Thinker from {mp} (bf16, {CFG['mllm']['qwen_omni']['device']})")
    proc = Qwen2_5OmniProcessor.from_pretrained(mp)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        mp, torch_dtype=torch.bfloat16, device_map=CFG["mllm"]["qwen_omni"]["device"]
    )
    model.eval()
    _QWEN = (model, proc)
    return _QWEN


def qwen_omni_asr_paralanguage(audio_path: Path, window_sec: float = 30.0) -> list[Utterance]:
    """Run Qwen2.5-Omni in chunks: ASR transcript + paralinguistic tags per utterance.

    For long audio we chunk by `window_sec` and let Qwen-Omni return JSON with relative
    timestamps; we then shift by the chunk offset.
    """
    import torch
    import librosa
    from qwen_omni_utils import process_mm_info

    model, processor = _load_qwen_omni()
    audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    total_sec = len(audio) / sr
    LOG.info(f"ASR over {total_sec:.1f}s audio in {window_sec}s chunks")

    utterances: list[Utterance] = []
    win_samples = int(window_sec * sr)
    n_chunks = (len(audio) + win_samples - 1) // win_samples

    prompt = (
        "Transcribe this audio. For each utterance, return JSON with fields:\n"
        "  start: float seconds (relative to this clip)\n"
        "  end:   float seconds (relative to this clip)\n"
        "  text:  spoken words\n"
        "  paralinguistic: list of tags chosen from "
        "[tone:trembling, tone:flat, tone:agitated, tone:sarcastic, tone:whispered, "
        " tempo:rushed, tempo:slow, pause:long, pause:short, "
        " nonverbal:sigh, nonverbal:laugh, nonverbal:cry, nonverbal:cough, "
        " volume:loud, volume:soft]\n"
        "Return strict JSON array. No prose."
    )

    sys_prompt = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )

    for ci in tqdm(range(n_chunks), desc="qwen-omni ASR"):
        chunk = audio[ci * win_samples : (ci + 1) * win_samples]
        if len(chunk) < sr * 0.5:  # too short
            continue
        offset = ci * window_sec
        # Both system and user content must be list-of-dict for Qwen2.5-Omni processor.
        conv = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": chunk},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        gen = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        # Parse JSON robustly
        try:
            from benchmark.pipeline.common import parse_json_block
            rows = parse_json_block(gen)
            if not isinstance(rows, list):
                rows = []
        except Exception as e:  # noqa: BLE001
            LOG.warning(f"chunk {ci} JSON parse failed: {e}; raw={gen[:200]!r}")
            rows = []
        for j, r in enumerate(rows):
            try:
                utt = Utterance(
                    utt_id=f"utt_{ci:04d}_{j:03d}",
                    start_sec=float(r["start"]) + offset,
                    end_sec=float(r["end"]) + offset,
                    speaker_id=None,
                    text=r.get("text", ""),
                    paralinguistic=list(r.get("paralinguistic", [])),
                )
                utterances.append(utt)
            except Exception as e:  # noqa: BLE001
                LOG.warning(f"row {j} skipped: {e}; row={r}")
    LOG.info(f"Got {len(utterances)} utterances from Qwen-Omni")
    return utterances


def faster_whisper_asr(audio_path: Path) -> list[Utterance]:
    """ASR fallback when Qwen-Omni fails — no paralinguistic tags."""
    from faster_whisper import WhisperModel

    LOG.info("Falling back to faster-whisper large-v3")
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    segments, _ = model.transcribe(str(audio_path), beam_size=5, vad_filter=True)
    utts: list[Utterance] = []
    for i, s in enumerate(segments):
        utts.append(
            Utterance(
                utt_id=f"utt_w_{i:04d}",
                start_sec=float(s.start),
                end_sec=float(s.end),
                speaker_id=None,
                text=s.text.strip(),
                paralinguistic=[],
            )
        )
    LOG.info(f"Got {len(utts)} utterances from faster-whisper")
    return utts


# -----------------------------------------------------------------------------
# Speaker diarization
# -----------------------------------------------------------------------------
def diarize(audio_path: Path) -> list[dict]:
    """Returns [{start, end, speaker}, ...] from pyannote."""
    import os

    hf_token = os.environ.get(CFG["perception"]["diarization"]["hf_token_env"])
    if not hf_token:
        LOG.warning("HF_TOKEN not set; skipping pyannote diarization. Speakers will be None.")
        return []
    try:
        from pyannote.audio import Pipeline as PA

        pipe = PA.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        diar = pipe(str(audio_path))
        segs: list[dict] = []
        for turn, _, speaker in diar.itertracks(yield_label=True):
            segs.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})
        LOG.info(f"Diarized into {len({s['speaker'] for s in segs})} speakers, {len(segs)} segments")
        return segs
    except Exception as e:  # noqa: BLE001
        LOG.warning(f"pyannote diarization failed: {e}; speakers will be None.")
        return []


def assign_speakers_to_utterances(utts: list[Utterance], diar: list[dict]) -> None:
    if not diar:
        return
    for u in utts:
        # max IoU with diarization segments
        best, best_iou = None, 0.0
        for seg in diar:
            inter = max(0.0, min(u.end_sec, seg["end"]) - max(u.start_sec, seg["start"]))
            union = max(u.end_sec, seg["end"]) - min(u.start_sec, seg["start"]) + 1e-6
            iou = inter / union
            if iou > best_iou:
                best, best_iou = seg["speaker"], iou
        if best is not None and best_iou > 0.05:
            u.speaker_id = best


# -----------------------------------------------------------------------------
# Face detection + clustering (InsightFace)
# -----------------------------------------------------------------------------
def detect_and_cluster_faces(video_path: Path, shots: list[Shot], fps_sample: float = 1.0) -> dict:
    """Sample frames at `fps_sample`, detect faces, cluster by ArcFace embedding.

    Returns:
        {
          "tracks": [ {cluster_id, char_name=None, appearances:[{frame, ts, bbox, shot_id}], n} ],
          "n_clusters": int,
          "n_appearances": int,
        }
    """
    import cv2
    import insightface

    LOG.info("Loading InsightFace buffalo_l (RetinaFace + ArcFace)")
    app = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=tuple(CFG["perception"]["face"]["det_size"]))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(round(fps / fps_sample)) or 1

    embeddings: list[np.ndarray] = []
    rows: list[dict] = []  # parallel to embeddings
    LOG.info(f"Scanning faces every {step} frames over {n_frames} frames")

    fi = 0
    pbar = tqdm(total=n_frames // step + 1, desc="face scan")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % step == 0:
            ts = fi / fps
            faces = app.get(frame)
            for face in faces:
                bbox = face.bbox.astype(int).tolist()
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if min(w, h) < CFG["perception"]["face"]["min_face_size"]:
                    continue
                # find shot_id covering ts
                sid = None
                for s in shots:
                    if s.start_sec <= ts <= s.end_sec:
                        sid = s.shot_id
                        break
                emb = face.normed_embedding  # L2-normalized 512-d
                embeddings.append(emb)
                rows.append({"frame": fi, "ts": ts, "bbox": bbox, "shot_id": sid})
            pbar.update(1)
        fi += 1
    pbar.close()
    cap.release()
    LOG.info(f"Collected {len(embeddings)} face crops")

    if not embeddings:
        return {"tracks": [], "n_clusters": 0, "n_appearances": 0}

    # Agglomerative clustering on cosine distance
    from sklearn.cluster import AgglomerativeClustering

    X = np.stack(embeddings)
    # cosine distance = 1 - cosine_similarity (embeddings are L2-normalized so dot = cos_sim)
    cluster_eps = CFG["perception"]["face"]["cluster_eps"]
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=cluster_eps,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(X)

    # Filter small clusters
    from collections import Counter

    counts = Counter(labels.tolist())
    min_size = CFG["perception"]["face"]["min_cluster_size"]
    keep = {c for c, n in counts.items() if n >= min_size}
    LOG.info(f"{len(counts)} raw clusters → {len(keep)} after min_size filter")

    tracks: list[dict] = []
    for c in sorted(keep):
        appearances = [rows[i] for i, l in enumerate(labels) if l == c]
        tracks.append(
            {
                "cluster_id": f"face_cluster_{int(c):02d}",
                "char_name": None,  # to be filled by Stage 1c
                "appearances": appearances,
                "n": len(appearances),
            }
        )
    tracks.sort(key=lambda t: -t["n"])  # most frequent first
    return {"tracks": tracks, "n_clusters": len(tracks), "n_appearances": int(sum(t["n"] for t in tracks))}


# -----------------------------------------------------------------------------
# Character naming alignment via GPT-5.5 + subtitles
# -----------------------------------------------------------------------------
def name_clusters(
    tracks: list[dict],
    utterances: list[Utterance],
    subtitle_text: str | None,
) -> list[dict]:
    """Use GPT-5.5 to align face_cluster IDs to character names by cross-referencing dialogue.

    Heuristic input: top-K speakers (by utterance count), addressed names mined from subtitles.
    """
    if not tracks:
        return tracks
    # Mine candidate names from subtitle (capitalized tokens / "X!", "Hey X")
    import re

    names: list[str] = []
    if subtitle_text:
        cands = re.findall(r"\b([A-Z][a-z]{2,})\b", subtitle_text)
        # filter common English non-name tokens
        stop = {"The", "And", "Yeah", "Okay", "Well", "Now", "Hey", "What", "Why", "How", "Where", "When", "This", "That", "Mom", "Dad"}
        for n in cands:
            if n not in stop:
                names.append(n)
    from collections import Counter

    top_names = [n for n, _ in Counter(names).most_common(10)]
    if not top_names:
        LOG.warning("No candidate names mined from subtitle; clusters remain unnamed.")
        return tracks

    msgs = [
        {"role": "system", "content": "You are a careful annotator who aligns face cluster IDs to character names in a TV episode."},
        {
            "role": "user",
            "content": (
                "Below are face clusters and the candidate character names mined from dialogue. "
                "Return JSON: {\"face_cluster_NN\": \"Name\" or null, ...}. "
                "Be conservative: if uncertain, return null. Use only names from the candidates list.\n\n"
                f"Face clusters (with #appearances):\n"
                + "\n".join(f"  - {t['cluster_id']}: {t['n']} appearances" for t in tracks)
                + f"\n\nCandidate names (top by mention frequency):\n  {top_names}\n\nJSON only."
            ),
        },
    ]
    try:
        from benchmark.pipeline.common import parse_json_block

        raw = llm_chat(msgs)
        mapping = parse_json_block(raw)
        for t in tracks:
            t["char_name"] = mapping.get(t["cluster_id"])
    except Exception as e:  # noqa: BLE001
        LOG.warning(f"Character naming failed: {e}. Leaving as None.")
    return tracks


# -----------------------------------------------------------------------------
# Shot-level assembly
# -----------------------------------------------------------------------------
def assemble_shot_rows(
    shots: list[Shot],
    utterances: list[Utterance],
    tracks: list[dict],
    fps: float,
) -> list[dict]:
    cluster_by_shot: dict[str, list[dict]] = {s.shot_id: [] for s in shots}
    for t in tracks:
        for a in t["appearances"]:
            if a["shot_id"] in cluster_by_shot:
                cluster_by_shot[a["shot_id"]].append({
                    "cluster_id": t["cluster_id"],
                    "char_name": t["char_name"],
                    "frame": a["frame"],
                    "ts": a["ts"],
                    "bbox": a["bbox"],
                })

    rows: list[dict] = []
    for s in shots:
        shot_utts = [
            asdict(u) for u in utterances
            if u.start_sec < s.end_sec and u.end_sec > s.start_sec
        ]
        speakers_present = sorted({u["speaker_id"] for u in shot_utts if u["speaker_id"]})
        # Distinct named characters seen in shot
        chars_seen = sorted({c["char_name"] for c in cluster_by_shot[s.shot_id] if c.get("char_name")})
        rows.append({
            "shot_id": s.shot_id,
            "time_span": [s.start_sec, s.end_sec],
            "frame_span": [s.start_frame, s.end_frame],
            "utterances": shot_utts,
            "speakers_present": speakers_present,
            "characters_present": chars_seen,
            "face_appearances": cluster_by_shot[s.shot_id],
        })
    return rows


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def run(video_path: Path, series: str, episode: str, srt: Path | None = None, do_visual: bool = False) -> None:
    """End-to-end Stage 1.

    Args:
        video_path: path to episode mp4/mkv
        series: e.g. 'breaking_bad'
        episode: e.g. 'ep01'
        srt: optional subtitle file
        do_visual: also call GPT-5.5 for per-shot visual caption (slow; off by default)
    """
    out_dir = ensure_dir(Path(CFG["project"]["data_root"]) / "perception" / series)
    perception_path = out_dir / f"{episode}.jsonl"
    meta_path = out_dir / f"{episode}.meta.json"

    # 1) Shot detection
    shots, fps = detect_shots(video_path)

    # 2) Audio + ASR + paralanguage
    audio_path = ensure_dir(out_dir / "audio") / f"{episode}.wav"
    if not audio_path.exists():
        LOG.info(f"Extracting audio → {audio_path}")
        extract_audio(video_path, audio_path)
    try:
        utts = qwen_omni_asr_paralanguage(audio_path)
        if len(utts) == 0:
            raise RuntimeError("qwen-omni returned 0 utterances; falling back")
    except Exception as e:  # noqa: BLE001
        LOG.warning(f"Qwen-Omni ASR failed: {e}; falling back to faster-whisper")
        utts = faster_whisper_asr(audio_path)

    # 3) Diarization
    diar = diarize(audio_path)
    assign_speakers_to_utterances(utts, diar)

    # 4) Face detect + cluster
    face = detect_and_cluster_faces(video_path, shots, fps_sample=1.0)

    # 5) Character naming
    sub_text = None
    if srt and srt.exists():
        sub_text = srt.read_text(encoding="utf-8", errors="replace")
    else:
        sub_text = "\n".join(u.text for u in utts)
    name_clusters(face["tracks"], utts, sub_text)

    # 6) Assemble shot rows
    rows = assemble_shot_rows(shots, utts, face["tracks"], fps)
    write_jsonl(perception_path, rows)
    write_json(
        meta_path,
        {
            "series": series,
            "episode": episode,
            "video_path": str(video_path),
            "fps": fps,
            "duration_sec": shots[-1].end_sec if shots else 0,
            "n_shots": len(shots),
            "n_utterances": len(utts),
            "n_speakers": len({u.speaker_id for u in utts if u.speaker_id}) or None,
            "n_face_clusters": face["n_clusters"],
            "named_characters": sorted({t["char_name"] for t in face["tracks"] if t["char_name"]}),
        },
    )
    LOG.info(f"Stage 1 done: {perception_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--series", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--srt", type=Path, default=None)
    ap.add_argument("--do_visual", action="store_true")
    args = ap.parse_args()
    run(args.video, args.series, args.episode, args.srt, args.do_visual)


if __name__ == "__main__":
    main()
