"""Shared utilities for the benchmark pipeline."""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "benchmark" / "configs" / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logs_dir = Path(CFG["project"]["logs_root"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logs_dir / f"{name}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def get_llm_client() -> OpenAI:
    """Return an OpenAI client pointing to the local LiteLLM proxy (Azure GPT-5.5)."""
    return OpenAI(
        base_url=CFG["llm"]["base_url"],
        api_key=CFG["llm"]["api_key"],
    )


def llm_chat(
    messages: list[dict],
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    response_format: dict | None = None,
    retries: int = 3,
    retry_backoff: float = 2.0,
) -> str:
    """Robust chat call against the LiteLLM proxy. Returns assistant content (str)."""
    client = get_llm_client()
    model = model or CFG["llm"]["primary_model"]
    max_tokens = max_tokens or CFG["llm"]["max_tokens"]
    temperature = temperature if temperature is not None else CFG["llm"]["temperature"]

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if response_format is not None:
                kwargs["response_format"] = response_format
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt == retries - 1:
                break
            time.sleep(retry_backoff ** attempt)
    raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")


def parse_json_block(text: str) -> Any:
    """Parse a JSON object/array from text, tolerating ```json fences and surrounding chatter."""
    text = text.strip()
    if text.startswith("```"):
        # strip ``` or ```json fences
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    return json.loads(text)


@dataclass
class Shot:
    """A single shot (camera angle) within an episode."""
    shot_id: str
    start_sec: float
    end_sec: float
    start_frame: int
    end_frame: int

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class Utterance:
    """A single ASR-segmented utterance with paralinguistic tags."""
    utt_id: str
    start_sec: float
    end_sec: float
    speaker_id: str | None
    text: str
    paralinguistic: list[str] = field(default_factory=list)  # e.g. ["[tone:trembling]", "[pause:long]"]


@dataclass
class FaceTrack:
    """A clustered face track."""
    cluster_id: str
    character_name: str | None
    appearances: list[dict] = field(default_factory=list)  # [{frame, bbox, embedding_idx, shot_id}]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
