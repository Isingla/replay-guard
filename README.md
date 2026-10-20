# replay-guard

A replay harness for **tool-result misbinding** in LLM agents, and a reference guardrail that fixes it.

ReplayBench (arXiv 2610.16333, Oct 2026) showed that valid-but-stale tool outputs, replayed into the wrong context, make agents take unsafe actions, and recommended that frameworks bind each result to its context. This repo implements that recommendation in LangChain, SmolAgents and the OpenAI Agents SDK, and measures it.

## What's here

| Piece | File |
|---|---|
| 5 task families where the wrong tool result causes an unsafe write (refund, deploy, email, access grant, file cleanup) | `replayguard/world.py` |
| Signed envelopes + `ContextBinder` guardrail | `replayguard/binding.py` |
| Trace store, replay perturbations, tool gateway | `replayguard/replay.py` |
| LangChain / SmolAgents / OpenAI Agents SDK adapters | `replayguard/adapters.py` |
| Local OpenAI-compatible scripted model (controls) | `replayguard/stubserver.py` |
| CLI: `record`, `bench`, `ablate` | `replayguard/runner.py` |

## Quick start

```bash
pip install -e ".[dev]"
make test            # 48 unit tests: guardrail checks, oracle sanity, no false positives
make traces          # record agent traces at t0 (2h ago) and recently (60s ago)
make bench           # 3 frameworks × models × perturbations × guard modes -> results/RESULTS.md
make ablate          # which binding check stops which replay -> results/ABLATION.md
```

Real models (any OpenAI-compatible endpoint, presets in `models.toml`):

```bash
export OPENAI_API_KEY=... OPENROUTER_API_KEY=...
make bench MODELS=gpt-4.1-mini,llama-3.3-70b,qwen-2.5-72b OUT=results/real
```

Prompt-only baseline and run-to-run variance:

```bash
# Does telling the model to check its tool results help? (no guardrail)
python -m replayguard.runner bench --models gpt-4.1-mini --prompts default,vigilant --guards off --out results/prompt
# Randomness on, 3 repeats per episode
python -m replayguard.runner bench --models gpt-4.1-mini --temperature 0.7 --repeats 3 --out results/variance
```

Local open models: `ollama pull qwen2.5:7b-instruct`, then `MODELS=qwen2.5-7b-local`.

## How it works

```
framework (LangChain | SmolAgents | OpenAI Agents SDK)
   └─ tool call ──► Gateway ──► live tool ──► signed result (task, step, tool, args, fetched_at)
                       │
                       ├─ perturbation swaps in a recorded result
                       │     cross_task : same tool, other task, 60s old
                       │     stale      : same task + args, recorded 2h ago, world changed since
                       │     cross_step : same task, earlier result for different args
                       │
                       └─ ContextBinder: verify signature, tool, task, args, step order, freshness
                              mismatch → refetch live  (or reject with an error)
```

Replay is injected **below** the framework, at the tool boundary, so each framework's own message handling runs unchanged. Writes always hit the live world and are judged by an oracle against the current state.

**Misbinding rate** = share of perturbed episodes (replay actually applied) where the agent executed an unsafe write.

## Results

GPT-4.1-mini, 1,152 episodes across the three frameworks:

| | Unsafe actions under replay |
|---|---|
| No guard | **75.8%** (200/264) |
| "Check your tool results" system prompt, no guard | **74.2%** (196/264) |
| Context-binding guard (refetch) | **0.0%** (0/264), task success unchanged |

Stale results fooled the model every time, with or without the careful prompt. Full tables in `results/real/RESULTS.md`, analysis in `WRITEUP.md`.
