# ForeScale results: reactive vs. predictive

Same seeded traffic curve (seed=42, day=180s), SLA = 500 ms p95.
Pod capacity = 50 rps, cold-start = 60s, safety margin = 30%.

| Metric | Reactive (HPA) | Predictive (ForeScale) |
|---|---:|---:|
| Max p95 latency (ms) | 9342 | 249 |
| Max p99 latency (ms) | 9792 | 293 |
| **Total SLA breach time (s)** | **20** | **0** |
| Requests over SLA | 1527 (9.45%) | 0 (0.00%) |
| Mean replicas | 7.5 | 3.9 |
| Peak replicas | 16 | 5 |

## Interpretation

- The reactive HPA breaches the SLA for **20s** because new pods take a full cold-start to become Ready *after* the load has already risen.
- ForeScale provisions 60s ahead, so pods are warm when the burst arrives: SLA breach time **0s**.
- ForeScale holds the SLA *and* runs cheaper on average (3.9 vs 7.5 mean replicas).

![comparison](comparison.png)
