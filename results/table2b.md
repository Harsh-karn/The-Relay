### Table 2b: Outcomes, same rows

| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
|---|---|---|---|---|---|---|---|
| steady | 33.3% | 66.7% | 0.0% | 0.33 | 1 | 30 | $0.00 |
| burst | 20.0% | 80.0% | 0.0% | 0.1 | 10 | 29 | $0.00 |
| sustained_overload | 11.4% | 60.0% | 28.6% | 0.11 | 10 | 29 | $0.00 |
| per_minute_limit_halved | 4.0% | 40.0% | 56.0% | 0.12 | 8 | 29 | $0.00 |
| daily_budget_exhausted | 0.0% | 100.0% | 0.0% | 0.0 | 1 | 21 | $0.00 |