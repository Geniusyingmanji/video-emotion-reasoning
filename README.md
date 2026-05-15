# Emotional Long-Video Reasoning Benchmark

End-to-end pipeline for constructing a 9-task × 4-evaluation-setting benchmark over
long-form TV content. Targets the "M1-M7 memory components × E0/E1/E2/E3 settings"
matrix described in `emotion_video_bench.md`.

See `survey.md` for the literature positioning and `emotion_video_bench.md` for the
design plan. See `SESSION_REPORT.md` for the latest end-to-end smoke-test
results.

## Quick start — 30 seconds to a working demo

```bash
conda activate emotion
cd /home/azureuser/workspace-gzy/zyf/video-emotion-reasoning

make test          # 4/4 smoke tests (parse_json_block, LLM proxy, Qwen-Omni files, paths)
make synth         # build a synthetic Breaking Bad scene perception (instant)
make synth-pipeline  # Stage 2 → 6: events, M2 DAG, trajectory, M3/M5/M6/M7, QA, filters (~10 min, all LLM-bound)
make synth-eval    # Stage 7: evaluate on 4 settings E0/E1/E2/E3 with GPT-5.5 (~5 min)
make report        # cross-setting comparison table
make preview-qa    # show the surviving QAs in human-readable form
make status        # what's been done across all series
```

Expected result on synthetic_demo: 6 surviving QAs after filters, E0 ~80% → E1 100%
(+~20pp local→episode gain) demonstrating the long-context-helps signal the
benchmark is designed to measure.

## Running on a real Breaking Bad episode

```bash
# 1. Drop episode mp4 (legitimately obtained — DVD/Blu-ray rip or paid digital
#    download) under data/raw/breaking_bad/s01/. Subtitles optional.
make stage1 VIDEO=data/raw/breaking_bad/s01/ep01.mp4 \
    SERIES=breaking_bad EPISODE=ep01 \
    CHARS='Walter Skyler Jesse Hank Marie' \
    SRT=data/raw/breaking_bad/s01/ep01.srt   # SRT optional but improves naming

# 2. Per-episode Stages 2-4
python -m benchmark.pipeline.run_pipeline single \
    --video data/raw/breaking_bad/s01/ep01.mp4 \
    --series breaking_bad --episode ep01

# 3. After 10 episodes have Stages 1-4 done:
make season SERIES=breaking_bad EPISODES='ep01 ep02 ep03 ep04 ep05 ep06 ep07 ep08 ep09 ep10'

# 4. Evaluate
python -m benchmark.pipeline.stage7_eval --series breaking_bad --episode ep01 --setting E0
python -m benchmark.pipeline.stage7_eval --series breaking_bad --episode ep01 --setting E1
python -m benchmark.pipeline.stage7_eval --series breaking_bad --episode ep01 --setting E2
python -m benchmark.pipeline.stage7_eval --series breaking_bad --episode ep01 --setting E3
make report SERIES=breaking_bad
```

## Setup

```bash
# 1. Use the pre-built emotion conda env
conda activate emotion
# python 3.11 + torch 2.5.1+cu121 + transformers 4.52.4 + Qwen2.5-Omni + scenedetect + pyannote + insightface

# 2. Verify model + LLM proxy
ls /home/azureuser/workspace-gzy/zyf/models/Qwen2.5-Omni-7B          # ~21G
curl -s http://localhost:4000/v1/models | head -5                      # LiteLLM proxy
```

If the env doesn't exist, recreate:
```bash
/opt/miniconda/condabin/conda create -n emotion python=3.11 pip -y
conda activate emotion
pip install -r requirements.txt
```

Critical pinning notes (see `requirements.txt` for why):
- `transformers==4.52.4` (Qwen2.5-Omni native support; later 5.x breaks torch 2.5)
- `torch==2.5.1+cu121` (needed for `ALL_PARALLEL_STYLES != None` in transformers)
- We bypass `Qwen2_5OmniForConditionalGeneration` and use `Qwen2_5OmniThinkerForConditionalGeneration`
  directly to avoid the `spk_dict.pt torch.load` that triggers CVE-2025-32434 (requires torch≥2.6).
- cuDNN is **disabled** at runtime (`torch.backends.cudnn.enabled = False`) because the cuDNN 9.2
  shipped with this PyTorch fails `CUDNN_STATUS_NOT_INITIALIZED` on conv ops. Conv falls back to
  non-cuDNN kernels at ~10-20% performance loss.

## Pipeline structure

```
benchmark/
├── configs/config.yaml          # central config
├── pipeline/
│   ├── common.py                # shared utils, LLM client, JSON parser
│   ├── stage1_perception.py     # shots + ASR + paralanguage + diarization + face cluster
│   ├── stage2_events.py         # affect/action/social event extraction (GPT-5.5 + Qwen-Omni)
│   ├── stage3_relations.py     # per-episode M2 DAG (intersection of two passes)
│   ├── stage4_trajectory.py     # per-episode M4 character emotion trajectory
│   ├── stage4_5_cross_episode.py # M3 cross-episode DAG + M4 cumulative
│   ├── stage4_6_semantic.py     # M5 OCEAN + M6 relations + M7 world state
│   ├── stage5_qgen.py           # 9-task QA generation
│   ├── stage6_filters.py        # F0a/b/c, F1, F5, F7, F8 quality filters
│   └── run_pipeline.py          # single-episode or season driver
├── prompts/
│   ├── event_extraction.md
│   ├── relation_extraction.md
│   ├── semantic_M{5,6,7}_*.md
│   └── qgen_T{1,2,4,5,6,7,8,9,10}.md
└── data/
    ├── raw/<series>/<episode>.mp4   # input videos (gitignored)
    ├── perception/<series>/<episode>.jsonl
    ├── events/<series>/<episode>.jsonl
    ├── event_graph/<series>/<episode>.M2.json
    ├── trajectory/<series>/<episode>/<character>.json
    ├── season/<series>/{M3,M4_*,M5_personas,M6_relations,M7_world}.json
    ├── qa/<series>/qa_staging.jsonl
    └── final/<series>/{qa_filtered,qa_audit}.jsonl
```

## Run the pipeline

```bash
# Single episode (Stages 1-4)
python -m benchmark.pipeline.run_pipeline single \
    --video data/raw/breaking_bad/s01/ep01.mp4 \
    --series breaking_bad --episode ep01

# Season-level merge (Stages 4.5-6) — requires Stages 1-4 done for every episode
python -m benchmark.pipeline.run_pipeline season \
    --series breaking_bad \
    --episodes ep01 ep02 ep03 ep04 ep05 ep06 ep07 ep08 ep09 ep10
```

Or run any single stage in isolation:
```bash
python -m benchmark.pipeline.stage1_perception --video V --series S --episode E
python -m benchmark.pipeline.stage2_events --series S --episode E
python -m benchmark.pipeline.stage3_relations --series S --episode E
python -m benchmark.pipeline.stage4_trajectory --series S --episode E
python -m benchmark.pipeline.stage4_5_cross_episode --series S --episodes ep01 ep02 ...
python -m benchmark.pipeline.stage4_6_semantic --series S --episodes ep01 ep02 ...
python -m benchmark.pipeline.stage5_qgen --series S --episodes ep01 --tasks T1 T2 T9
python -m benchmark.pipeline.stage6_filters --series S --in_qa data/qa/S/qa_staging.jsonl
```

## Data acquisition

The plan calls for 3 English drama series × 10 episodes each (Breaking Bad, Mad Men,
The Sopranos as the recommended set). Video files are **NOT committed**; users supply
their own via legitimate channels (DVD/Blu-ray rip of owned copies, licensed streaming
download, etc.) and drop them under `data/raw/<series>/`.

For development/code testing, we use `data/raw/breaking_bad/dev/sintel_cc_by.mkv`
(Sintel, Blender Foundation CC-BY) and `data/raw/breaking_bad/dev/tos_4min_720p.mp4`
(Tears of Steel, Blender Foundation CC-BY, 4-min slice). Both are 100% legitimate and
suitable for code verification.

## Known issues

- **Qwen2.5-Omni open-ended JSON ASR is unreliable**: short utterances produce
  hallucinated content. We use **faster-whisper for ASR** and **Qwen-Omni for
  paralanguage tagging on already-segmented utterances** (constrained task, more reliable).
- **InsightFace ONNX runtime defaults to CPU** in this env (the `onnxruntime-gpu`
  package didn't register the CUDA provider). Face detection runs on CPU at ~3 fps;
  fine for our 1 fps sampling.
- **pyannote diarization is gated**: set `HF_TOKEN` env var with an HF account that
  has accepted the `pyannote/speaker-diarization-3.1` agreement. Without it, speaker
  IDs remain `None`.

## Roadmap

See `emotion_video_bench.md` Verification 路线 section and the GitHub Issues.
