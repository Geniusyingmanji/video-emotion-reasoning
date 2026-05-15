"""Smoke tests for the pipeline. Designed to fail fast and inform.

Run:
  python -m benchmark.tests.test_pipeline_smoke
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.pipeline.common import CFG, parse_json_block, get_llm_client  # noqa: E402


def test_parse_json_block() -> None:
    cases = [
        ('[{"a":1}]', [{"a": 1}]),
        ('```json\n[{"a":1}]\n```', [{"a": 1}]),
        ('Sure:\n[{"x":1},{"x":2}]\nThanks!', [{"x": 1}, {"x": 2}]),
        ('[{"a":"b]c"}]', [{"a": "b]c"}]),
        ('  [ {"a":1} ] \nHuman: garbage', [{"a": 1}]),
    ]
    for raw, expected in cases:
        got = parse_json_block(raw)
        assert got == expected, f"mismatch for {raw!r}: got {got}, expected {expected}"
    print("  parse_json_block: 5/5 PASS")


def test_llm_proxy() -> None:
    client = get_llm_client()
    # GPT-5.5 routes through reasoning; need ≥20 max_tokens for the assistant content
    # to come back non-empty (reasoning tokens are not counted in 'content').
    resp = client.chat.completions.create(
        model=CFG["llm"]["primary_model"],
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        max_tokens=64,
        temperature=0.0,
    )
    out = (resp.choices[0].message.content or "").strip().lower()
    assert "ok" in out, f"unexpected: {out!r}"
    print(f"  LLM proxy ({CFG['llm']['primary_model']}): PASS ({out!r})")


def test_config_paths() -> None:
    for k in ("data_root", "models_root", "logs_root"):
        p = Path(CFG["project"][k])
        assert p.exists() or p.parent.exists(), f"{k} -> {p} parent doesn't exist"
    print("  config paths: PASS")


def test_qwen_omni_model_files() -> None:
    p = Path(CFG["mllm"]["qwen_omni"]["path"])
    assert p.exists(), f"Qwen2.5-Omni dir missing: {p}"
    shards = sorted(p.glob("model-*.safetensors"))
    assert len(shards) >= 4, f"expected ≥4 shards, found {len(shards)}"
    print(f"  Qwen2.5-Omni files: PASS ({len(shards)} shards)")


def main() -> None:
    print("Running smoke tests...")
    for fn in (test_config_paths, test_parse_json_block, test_qwen_omni_model_files, test_llm_proxy):
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print("Smoke tests done.")


if __name__ == "__main__":
    main()
