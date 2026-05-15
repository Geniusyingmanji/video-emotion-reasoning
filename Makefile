.PHONY: help test smoke synth-pipeline synth-eval clean

PY := /home/azureuser/.conda/envs/emotion/bin/python
SERIES ?= synthetic_demo
EPISODE ?= ep01_demo

help:
	@echo "Targets:"
	@echo "  test           Run smoke test (parse_json_block, LLM proxy, model files, paths)"
	@echo "  synth          Make synthetic perception fixture"
	@echo "  synth-pipeline Stage 2 -> 6 on synthetic_demo/ep01_demo"
	@echo "  synth-eval     Stage 7 evaluation for all E0/E1/E2/E3 on synthetic"
	@echo "  stage1 VIDEO=/path/to/X.mp4 SERIES=Y EPISODE=Z  Run Stage 1 on a video"
	@echo "  season SERIES=Y EPISODES='ep01 ep02 ...'  Run Stages 4.5 -> 6 for a season"
	@echo "  report SERIES=Y  Cross-setting E0-E3 accuracy comparison table"
	@echo "  status [SERIES=Y]  Per-series stage state and recommended next action"
	@echo "  preview-qa SERIES=Y  Human-readable view of surviving QAs"
	@echo "  clean-synth    Remove synthetic_demo outputs (keeps raw videos and models)"

test:
	$(PY) -m benchmark.tests.test_pipeline_smoke

synth:
	$(PY) -m benchmark.scripts.make_synthetic_perception

synth-pipeline: synth
	$(PY) -m benchmark.pipeline.stage2_events --series $(SERIES) --episode $(EPISODE) --skip_passB
	$(PY) -m benchmark.pipeline.stage3_relations --series $(SERIES) --episode $(EPISODE)
	$(PY) -m benchmark.pipeline.stage4_trajectory --series $(SERIES) --episode $(EPISODE)
	$(PY) -m benchmark.pipeline.stage4_5_cross_episode --series $(SERIES) --episodes $(EPISODE)
	$(PY) -m benchmark.pipeline.stage4_6_semantic --series $(SERIES) --episodes $(EPISODE)
	$(PY) -m benchmark.pipeline.stage5_qgen --series $(SERIES) --episodes $(EPISODE) \
	    --tasks T1 T2 T4 T6 T7 T9 T10 --max_per_task 4
	$(PY) -m benchmark.pipeline.stage6_filters --series $(SERIES) \
	    --in_qa data/qa/$(SERIES)/qa_staging.jsonl

synth-eval:
	$(PY) -m benchmark.pipeline.stage7_eval --series $(SERIES) --episode $(EPISODE) --setting E0
	$(PY) -m benchmark.pipeline.stage7_eval --series $(SERIES) --episode $(EPISODE) --setting E1
	$(PY) -m benchmark.pipeline.stage7_eval --series $(SERIES) --episode $(EPISODE) --setting E2
	$(PY) -m benchmark.pipeline.stage7_eval --series $(SERIES) --episode $(EPISODE) --setting E3

synth-eval-qwen:
	$(PY) -m benchmark.pipeline.stage7_eval_qwen_omni --series $(SERIES) --episode $(EPISODE) --setting E0
	$(PY) -m benchmark.pipeline.stage7_eval_qwen_omni --series $(SERIES) --episode $(EPISODE) --setting E1

report:
	$(PY) -m benchmark.scripts.report_eval --series $(SERIES)

status:
	$(PY) -m benchmark.scripts.status

preview-qa:
	$(PY) -m benchmark.scripts.preview_qa --series $(SERIES) --limit 30

stage1:
	@[ -n "$(VIDEO)" ] || (echo "Usage: make stage1 VIDEO=/path/X.mp4 SERIES=Y EPISODE=Z [CHARS='Walter Skyler']"; exit 1)
	$(PY) -m benchmark.pipeline.stage1_perception \
	    --video $(VIDEO) --series $(SERIES) --episode $(EPISODE) \
	    $(if $(CHARS),--characters $(CHARS),) $(if $(SRT),--srt $(SRT),)

season:
	@[ -n "$(EPISODES)" ] || (echo "Usage: make season SERIES=Y EPISODES='ep01 ep02 ...'"; exit 1)
	$(PY) -m benchmark.pipeline.stage4_5_cross_episode --series $(SERIES) --episodes $(EPISODES)
	$(PY) -m benchmark.pipeline.stage4_6_semantic --series $(SERIES) --episodes $(EPISODES)
	$(PY) -m benchmark.pipeline.stage5_qgen --series $(SERIES) --episodes $(EPISODES)
	$(PY) -m benchmark.pipeline.stage6_filters --series $(SERIES) \
	    --in_qa data/qa/$(SERIES)/qa_staging.jsonl

clean-synth:
	rm -rf data/perception/synthetic_demo data/events/synthetic_demo \
	       data/event_graph/synthetic_demo data/trajectory/synthetic_demo \
	       data/season/synthetic_demo data/qa/synthetic_demo \
	       data/final/synthetic_demo data/eval/synthetic_demo
	@echo "Cleaned synthetic_demo outputs"
