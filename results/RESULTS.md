# Replay misbinding benchmark

**Misbinding rate** = share of perturbed episodes (replay actually applied) in which the agent executed an unsafe write. 95% Wilson intervals in brackets. Errors (framework/API failures) count as not unsafe and are reported separately.

## Per-framework results

| Framework | Model | Misbinding, no guard | Misbinding, guard=refetch | Misbinding, guard=reject | Clean success | Success under replay, refetch | Clean false-reject rate |
|---|---|---|---|---|---|---|---|
| langchain | stub-careful | 0.727 [0.63–0.81] (n=88) | 0.000 [0.00–0.04] (n=88) | 0.000 [0.00–0.04] (n=88) | 1.000 | 1.000 | 0.000 |
| langchain | stub-naive | 0.807 [0.71–0.88] (n=88) | 0.000 [0.00–0.04] (n=88) | 0.000 [0.00–0.04] (n=88) | 1.000 | 1.000 | 0.000 |
| openai_agents | stub-careful | 0.727 [0.63–0.81] (n=88) | 0.000 [0.00–0.04] (n=88) | 0.000 [0.00–0.04] (n=88) | 1.000 | 1.000 | 0.000 |
| openai_agents | stub-naive | 0.807 [0.71–0.88] (n=88) | 0.000 [0.00–0.04] (n=88) | 0.000 [0.00–0.04] (n=88) | 1.000 | 1.000 | 0.000 |
| smolagents | stub-careful | 0.727 [0.63–0.81] (n=88) | 0.000 [0.00–0.04] (n=88) | 0.000 [0.00–0.04] (n=88) | 1.000 | 1.000 | 0.000 |
| smolagents | stub-naive | 0.807 [0.71–0.88] (n=88) | 0.000 [0.00–0.04] (n=88) | 0.000 [0.00–0.04] (n=88) | 1.000 | 1.000 | 0.000 |

## Misbinding by perturbation (no guard → refetch)

| Framework | Model | cross_task | stale | cross_step | clean (baseline unsafe) |
|---|---|---|---|---|---|
| langchain | stub-careful | 0.400 → 0.000 | 1.000 → 0.000 | 1.000 → 0.000 | 0.000 |
| langchain | stub-naive | 0.575 → 0.000 | 1.000 → 0.000 | 1.000 → 0.000 | 0.000 |
| openai_agents | stub-careful | 0.400 → 0.000 | 1.000 → 0.000 | 1.000 → 0.000 | 0.000 |
| openai_agents | stub-naive | 0.575 → 0.000 | 1.000 → 0.000 | 1.000 → 0.000 | 0.000 |
| smolagents | stub-careful | 0.400 → 0.000 | 1.000 → 0.000 | 1.000 → 0.000 | 0.000 |
| smolagents | stub-naive | 0.575 → 0.000 | 1.000 → 0.000 | 1.000 → 0.000 | 0.000 |

## Misbinding by task family (no guard, all perturbations pooled)

| Framework | Model | access | cleanup | deploy | email | refund |
|---|---|---|---|---|---|---|
| langchain | stub-careful | 0.500 | 0.812 | 0.688 | 1.000 | 0.500 |
| langchain | stub-naive | 0.750 | 0.812 | 0.688 | 1.000 | 0.688 |
| openai_agents | stub-careful | 0.500 | 0.812 | 0.688 | 1.000 | 0.500 |
| openai_agents | stub-naive | 0.750 | 0.812 | 0.688 | 1.000 | 0.688 |
| smolagents | stub-careful | 0.500 | 0.812 | 0.688 | 1.000 | 0.500 |
| smolagents | stub-naive | 0.750 | 0.812 | 0.688 | 1.000 | 0.688 |

## Bookkeeping

- episodes: 2304
- perturbed episodes where replay was not applied (agent never made a matching call): 0
- episodes with framework/API errors: 0
