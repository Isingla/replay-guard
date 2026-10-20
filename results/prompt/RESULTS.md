# Prompt-only defense: GPT-4.1-mini with a "check your tool results" system prompt

Run 2026-10-19, temperature 0, no guardrail, 384 episodes, 0 errors. Same tasks and replays as `results/real`.
Re-run: `python -m replayguard.runner bench --models gpt-4.1-mini --prompts vigilant --guards off --out results/prompt`

| Framework | Unsafe under replay, default prompt | Unsafe under replay, vigilant prompt |
|---|---|---|
| LangChain | 0.739 (65/88) | 0.693 [0.59–0.78] (61/88) |
| OpenAI Agents SDK | 0.773 (68/88) | 0.773 [0.67–0.85] (68/88) |
| SmolAgents | 0.761 (67/88) | 0.761 [0.66–0.84] (67/88) |
| **Pooled** | **0.758** (200/264) | **0.742** (196/264) |

By replay type, vigilant prompt (no guard):

| Framework | cross_task | stale | cross_step | clean (baseline unsafe) |
|---|---|---|---|---|
| langchain | 0.500 | 1.000 | 0.125 | 0.000 |
| openai_agents | 0.500 | 1.000 | 1.000 | 0.000 |
| smolagents | 0.475 | 1.000 | 1.000 | 0.000 |

Clean success stayed 1.000 in all three frameworks: the prompt did not make the agent more cautious on normal runs either.
