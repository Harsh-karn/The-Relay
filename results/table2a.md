### Table 2a: Latency, one row per load scenario

| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
|---|---|---|---|---|---|---|
| steady | stand-in | 15 | 34.6 | 26.0 | 20 / 129 / 157 | 9 / 1584 / 2365 |
| burst | stand-in | 20 | 1.0 | 1200.0 | 251 / 271 / 271 | 30 / 464 / 710 |
| sustained_overload | stand-in | 35 | 15.5 | 135.5 | 20 / 20 / 20 | 7 / 12016 / 12165 |
| per_minute_limit_halved | stand-in | 25 | 29.1 | 51.6 | 0 / 18 / 20 | 12013 / 12019 / 12230 |
| daily_budget_exhausted | stand-in | 15 | 0.1 | 9000.0 | 0 / 0 / 0 | 1 / 1 / 1 |