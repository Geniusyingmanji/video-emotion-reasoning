"""Quick inspector for Stage 1 output. Prints per-shot summary and meta."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.pipeline.common import CFG, read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--head", type=int, default=10)
    args = ap.parse_args()

    base = Path(CFG["project"]["data_root"]) / "perception" / args.series
    rows = read_jsonl(base / f"{args.episode}.jsonl")
    meta = json.loads((base / f"{args.episode}.meta.json").read_text())

    print(f"=== {args.series}/{args.episode} ===")
    print(json.dumps(meta, indent=2))
    print()
    print(f"First {args.head} shots:")
    for r in rows[: args.head]:
        ts = r["time_span"]
        nu = len(r["utterances"])
        chars = ",".join(r["characters_present"]) or "?"
        nf = len(r["face_appearances"])
        print(f"  {r['shot_id']:>10} [{ts[0]:7.2f}-{ts[1]:7.2f}] utts={nu:2d} faces={nf:3d} chars={chars}")
        for u in r["utterances"][:3]:
            tags = " ".join(u.get("paralinguistic", []))
            print(f"     {u['start_sec']:.2f}-{u['end_sec']:.2f} {u.get('speaker_id') or '?'}: {u['text']!r}  {tags}")

    total_utts = sum(len(r["utterances"]) for r in rows)
    total_faces = sum(len(r["face_appearances"]) for r in rows)
    print()
    print(f"Totals: {len(rows)} shots, {total_utts} utts, {total_faces} face apps")


if __name__ == "__main__":
    main()
