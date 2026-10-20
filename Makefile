PY ?= python
N ?= 8
MODELS ?= stub-naive,stub-careful
FRAMEWORKS ?= langchain,smolagents,openai_agents
RECORDER ?= stub-naive
OUT ?= results

.PHONY: install test traces bench ablate all
install:
	$(PY) -m pip install -e ".[dev]"
test:
	$(PY) -m pytest -q
traces:   ## record agent traces at t0 (2h ago) and recently (60s ago)
	$(PY) -m replayguard.runner record --framework langchain --model $(RECORDER) --n $(N) --out traces/traces.jsonl
bench:
	$(PY) -m replayguard.runner bench --n $(N) --models $(MODELS) --frameworks $(FRAMEWORKS) --out $(OUT)
ablate:
	$(PY) -m replayguard.runner ablate --n $(N) --model stub-naive --out $(OUT)
all: test traces bench ablate
