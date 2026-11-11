### Table 2a: Latency, one row per load scenario

| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
|---|---|---|---|---|---|---|
| steady | stand-in | 15 | 34.8 | 25.9 | 20 / 179 / 210 | 5 / 1472 / 2603 |
| burst | stand-in | 20 | 1.0 | 1200.0 | 250 / 271 / 271 | 11 / 509 / 648 |
| sustained_overload | stand-in | 35 | 15.4 | 136.1 | 20 / 20 / 20 | 6 / 12012 / 12012 |
| per_minute_limit_halved | stand-in | 25 | 29.0 | 51.7 | 0 / 0 / 0 | 12011 / 12013 / 12013 |
| daily_budget_exhausted | stand-in | 15 | 0.1 | 9000.0 | 0 / 0 / 0 | 1 / 1 / 1 |