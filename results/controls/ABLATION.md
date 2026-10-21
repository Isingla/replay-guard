# Guardrail ablation

Framework `langchain`, model `stub-naive`, mode=refetch. Cells are misbinding rates over episodes where replay was applied.

| Checks enabled | cross_task | stale | cross_step | pooled |
|---|---|---|---|---|
| none | 0.575 | 1.000 | 1.000 | 0.807 |
| task only | 0.000 | 1.000 | 1.000 | 0.545 |
| freshness only | 0.575 | 0.000 | 1.000 | 0.352 |
| args+step only | 0.000 | 1.000 | 0.000 | 0.455 |
| full | 0.000 | 0.000 | 0.000 | 0.000 |
