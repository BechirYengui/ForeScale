# ForeScale results: reactive vs. predictive

Same seeded traffic curve (seed=42, day=600s), SLA = 500 ms p95.
Pod capacity = 50 rps, cold-start = 60s, safety margin = 30%.

| Metric | Reactive (HPA) | Predictive (ForeScale) |
|---|---:|---:|
| Max p95 latency (ms) | 9848 | 283 |
| Max p99 latency (ms) | 11169 | 331 |
| **Total SLA breach time (s)** | **110** | **0** |
| Requests over SLA | 12861 (20.43%) | 0 (0.00%) |
| Mean replicas | 3.9 | 4.2 |
| Peak replicas | 8 | 5 |

## Interpretation

- The reactive HPA breaches the SLA for **110s** because new pods take a full cold-start to become Ready *after* the load has already risen.
- ForeScale provisions 60s ahead, so pods are warm when the burst arrives: SLA breach time **0s**.
- ForeScale holds the SLA at a comparable average pod cost (4.2 vs 3.9 mean replicas) -- it spends its capacity *when it matters* instead of as standing slack.

![comparison](comparison.png)
