"""Generate a synthetic perception trace + meta for end-to-end testing of Stage 2-6.

Hand-crafted to mimic an early Breaking Bad scene with realistic dialogue, character
emotions, and shot timing. Lets us exercise the LLM-driven pipeline (Stage 2-6) without
waiting for the (slow CPU) Stage 1 face scan to finish.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.pipeline.common import CFG, ensure_dir, write_json, write_jsonl  # noqa: E402

SERIES = "synthetic_demo"
EPISODE = "ep01_demo"

# Synthetic perception ~4 min, two characters (Walter, Skyler), short scene.
SHOTS = [
    # shot_id, start, end, utterances, characters_present
    ("shot_0000", 0.0, 8.0, [], ["Walter"]),
    ("shot_0001", 8.0, 22.0, [
        {"utt_id": "utt_0000", "start_sec": 8.5, "end_sec": 12.0, "speaker_id": "SPEAKER_00", "text": "I have something to tell you, Skyler.", "paralinguistic": ["tone:flat", "tempo:slow"]},
        {"utt_id": "utt_0001", "start_sec": 13.0, "end_sec": 16.0, "speaker_id": "SPEAKER_01", "text": "What is it?", "paralinguistic": []},
        {"utt_id": "utt_0002", "start_sec": 16.5, "end_sec": 21.5, "speaker_id": "SPEAKER_00", "text": "I went to the doctor. It's cancer. Stage three.", "paralinguistic": ["tone:trembling", "pause:long"]},
    ], ["Walter", "Skyler"]),
    ("shot_0002", 22.0, 35.0, [
        {"utt_id": "utt_0003", "start_sec": 24.0, "end_sec": 26.5, "speaker_id": "SPEAKER_01", "text": "Oh my god.", "paralinguistic": ["tone:trembling", "volume:soft"]},
        {"utt_id": "utt_0004", "start_sec": 28.0, "end_sec": 33.5, "speaker_id": "SPEAKER_01", "text": "But you—you'll get treatment, right? We'll fight this.", "paralinguistic": ["tone:agitated", "tempo:rushed"]},
    ], ["Skyler"]),
    ("shot_0003", 35.0, 50.0, [
        {"utt_id": "utt_0005", "start_sec": 36.0, "end_sec": 42.0, "speaker_id": "SPEAKER_00", "text": "The treatment costs ninety thousand dollars. We don't have it.", "paralinguistic": ["tone:flat"]},
        {"utt_id": "utt_0006", "start_sec": 43.0, "end_sec": 48.0, "speaker_id": "SPEAKER_01", "text": "We'll find a way. Don't you give up on me.", "paralinguistic": ["tone:agitated", "volume:loud"]},
    ], ["Walter", "Skyler"]),
    ("shot_0004", 50.0, 70.0, [
        {"utt_id": "utt_0007", "start_sec": 52.0, "end_sec": 55.0, "speaker_id": "SPEAKER_00", "text": "I'm sorry.", "paralinguistic": ["tone:trembling", "nonverbal:sigh"]},
    ], ["Walter"]),
    ("shot_0005", 70.0, 95.0, [
        {"utt_id": "utt_0008", "start_sec": 72.0, "end_sec": 76.0, "speaker_id": "SPEAKER_00", "text": "Jesse, are you there? It's Mr. White.", "paralinguistic": ["tone:flat"]},
        {"utt_id": "utt_0009", "start_sec": 78.0, "end_sec": 84.0, "speaker_id": "SPEAKER_02", "text": "Mr. White? Why are you calling me?", "paralinguistic": ["tone:agitated"]},
        {"utt_id": "utt_0010", "start_sec": 85.0, "end_sec": 91.0, "speaker_id": "SPEAKER_00", "text": "I want in. I want to cook with you.", "paralinguistic": ["tone:flat", "tempo:slow"]},
    ], ["Walter"]),
    ("shot_0006", 95.0, 115.0, [
        {"utt_id": "utt_0011", "start_sec": 96.0, "end_sec": 100.0, "speaker_id": "SPEAKER_02", "text": "Are you serious right now?", "paralinguistic": ["tone:agitated", "volume:loud"]},
        {"utt_id": "utt_0012", "start_sec": 102.0, "end_sec": 110.0, "speaker_id": "SPEAKER_00", "text": "You know the business. I know the chemistry. We can make money.", "paralinguistic": ["tone:flat"]},
    ], ["Jesse"]),
    ("shot_0007", 115.0, 140.0, [
        {"utt_id": "utt_0013", "start_sec": 117.0, "end_sec": 122.0, "speaker_id": "SPEAKER_02", "text": "Fine. But this is my world now. You play by my rules.", "paralinguistic": ["tone:agitated"]},
        {"utt_id": "utt_0014", "start_sec": 124.0, "end_sec": 128.0, "speaker_id": "SPEAKER_00", "text": "Whatever it takes.", "paralinguistic": ["tone:flat"]},
    ], ["Walter", "Jesse"]),
    ("shot_0008", 140.0, 165.0, [
        {"utt_id": "utt_0015", "start_sec": 142.0, "end_sec": 148.0, "speaker_id": "SPEAKER_01", "text": "Walt? Where have you been all day?", "paralinguistic": []},
        {"utt_id": "utt_0016", "start_sec": 150.0, "end_sec": 156.0, "speaker_id": "SPEAKER_00", "text": "Just driving. Thinking. I needed air.", "paralinguistic": ["tone:flat"]},
    ], ["Walter", "Skyler"]),
    ("shot_0009", 165.0, 200.0, [
        {"utt_id": "utt_0017", "start_sec": 168.0, "end_sec": 175.0, "speaker_id": "SPEAKER_01", "text": "I called Hank. He says he can help us with the bills.", "paralinguistic": []},
        {"utt_id": "utt_0018", "start_sec": 178.0, "end_sec": 186.0, "speaker_id": "SPEAKER_00", "text": "No! Don't tell Hank anything. I forbid it.", "paralinguistic": ["tone:agitated", "volume:loud"]},
        {"utt_id": "utt_0019", "start_sec": 188.0, "end_sec": 195.0, "speaker_id": "SPEAKER_01", "text": "Walt, what is going on with you?", "paralinguistic": ["tone:agitated"]},
    ], ["Walter", "Skyler"]),
    ("shot_0010", 200.0, 240.0, [
        {"utt_id": "utt_0020", "start_sec": 202.0, "end_sec": 208.0, "speaker_id": "SPEAKER_00", "text": "I'm a dying man trying to leave something for his family. That's all.", "paralinguistic": ["tone:flat", "tempo:slow"]},
        {"utt_id": "utt_0021", "start_sec": 215.0, "end_sec": 222.0, "speaker_id": "SPEAKER_01", "text": "Then let us help you. Don't shut me out.", "paralinguistic": ["tone:trembling", "volume:soft"]},
    ], ["Walter", "Skyler"]),
]


def main() -> None:
    data_root = Path(CFG["project"]["data_root"])
    out_dir = ensure_dir(data_root / "perception" / SERIES)

    rows = []
    for shot_id, start, end, utts, chars in SHOTS:
        rows.append({
            "shot_id": shot_id,
            "time_span": [start, end],
            "frame_span": [int(start * 24), int(end * 24)],
            "utterances": utts,
            "speakers_present": sorted({u["speaker_id"] for u in utts if u.get("speaker_id")}),
            "characters_present": chars,
            "face_appearances": [{"cluster_id": f"face_cluster_{i:02d}",
                                   "char_name": c,
                                   "frame": int((start + i) * 24),
                                   "ts": start + i,
                                   "bbox": [100 + i * 10, 100, 300, 300]}
                                  for i, c in enumerate(chars)],
        })

    perception_path = out_dir / f"{EPISODE}.jsonl"
    meta_path = out_dir / f"{EPISODE}.meta.json"
    write_jsonl(perception_path, rows)
    duration = SHOTS[-1][2]
    write_json(meta_path, {
        "series": SERIES, "episode": EPISODE,
        "video_path": "synthetic", "fps": 24.0,
        "duration_sec": duration,
        "n_shots": len(SHOTS),
        "n_utterances": sum(len(r["utterances"]) for r in rows),
        "n_speakers": 3,
        "n_face_clusters": 3,
        "named_characters": sorted({c for _, _, _, _, chars in SHOTS for c in chars}),
    })

    # Synthetic 16kHz silent WAV (so audio_path exists for any future steps)
    audio_path = ensure_dir(out_dir / "audio") / f"{EPISODE}.wav"
    if not audio_path.exists():
        import numpy as np, soundfile as sf
        n = int(duration * 16000)
        sf.write(str(audio_path), np.zeros(n, dtype=np.float32), 16000)

    print(f"Wrote {perception_path}")
    print(f"Wrote {meta_path}")
    print(f"Synthetic perception: {len(SHOTS)} shots, {sum(len(r['utterances']) for r in rows)} utts, "
          f"{duration:.1f}s, 3 characters (Walter, Skyler, Jesse)")


if __name__ == "__main__":
    main()
