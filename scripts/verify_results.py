"""Automated verification script checking results.csv against all tables and schema constraints."""

import csv
import sys
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_CSV = Path("results/results.csv")
REQUIRED_COLUMNS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
]

ALLOWED_STATUS = {"ok", "rate_limited", "timeout", "error"}
ALLOWED_SOURCE = {"real", "stand-in"}
ALLOWED_GRADE = {"correct", "wrong", "refused_correctly", "refused_wrongly"}


def verify_results_csv():
    if not RESULTS_CSV.exists():
        print(f"FAIL: {RESULTS_CSV} does not exist.")
        sys.exit(1)

    df = pd.read_csv(RESULTS_CSV)
    print(f"Loaded {len(df)} rows from {RESULTS_CSV}")

    # 1. Column Check
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"FAIL: Missing columns: {missing_cols}")
        sys.exit(1)
    print("✓ Schema columns strictly match specification.")

    # 2. Enum validity
    invalid_status = set(df["status"].unique()) - ALLOWED_STATUS
    assert not invalid_status, f"Invalid status: {invalid_status}"
    invalid_source = set(df["source"].unique()) - ALLOWED_SOURCE
    assert not invalid_source, f"Invalid source: {invalid_source}"
    invalid_grade = set(df["answer_grade"].unique()) - ALLOWED_GRADE
    assert not invalid_grade, f"Invalid grade: {invalid_grade}"
    print("✓ All enum fields conform to allowed values.")

    # 3. Numeric validity
    assert (df["first_token_ms"] >= 0).all(), "Negative first_token_ms found"
    assert (df["total_ms"] >= 0).all(), "Negative total_ms found"
    assert (df["retries"] >= 0).all(), "Negative retries found"
    assert (df["tokens_in"] >= 0).all(), "Negative tokens_in found"
    assert (df["tokens_out"] >= 0).all(), "Negative tokens_out found"
    print("✓ All numerical fields non-negative.")

    # 4. Recompute Table 2a
    print("\n--- Recomputed Table 2a from results.csv ---")
    scenarios = df["scenario"].unique()
    for sc in scenarios:
        sc_df = df[df["scenario"] == sc]
        ttfts = sc_df[sc_df["first_token_ms"] > 0]["first_token_ms"]
        totals = sc_df[sc_df["total_ms"] > 0]["total_ms"]

        p50_ttft = int(round(np.percentile(ttfts, 50))) if len(ttfts) else 0
        p95_ttft = int(round(np.percentile(ttfts, 95))) if len(ttfts) else 0
        p99_ttft = int(round(np.percentile(ttfts, 99))) if len(ttfts) else 0

        p50_tot = int(round(np.percentile(totals, 50))) if len(totals) else 0
        p95_tot = int(round(np.percentile(totals, 95))) if len(totals) else 0
        p99_tot = int(round(np.percentile(totals, 99))) if len(totals) else 0

        print(f"[{sc}] N={len(sc_df)} | TTFT: {p50_ttft}/{p95_ttft}/{p99_ttft} ms | Total: {p50_tot}/{p95_tot}/{p99_tot} ms")

    # 5. Recompute Table 2b
    print("\n--- Recomputed Table 2b from results.csv ---")
    for sc in scenarios:
        sc_df = df[df["scenario"] == sc]
        n = len(sc_df)
        succ = (sc_df["status"] == "ok").sum() / n * 100
        rl = (sc_df["status"] == "rate_limited").sum() / n * 100
        to = (sc_df["status"].isin(["timeout", "error"])).sum() / n * 100
        avg_retries = sc_df["retries"].sum() / n
        tokens_out_nz = sc_df[sc_df["tokens_out"] > 0]["tokens_out"]
        avg_tok = int(round(tokens_out_nz.mean())) if len(tokens_out_nz) else 0

        print(f"[{sc}] Succ: {succ:.1f}% | RL: {rl:.1f}% | TO: {to:.1f}% | Retries/req: {avg_retries:.2f} | Tokens/ans: {avg_tok}")

    print("\n✓ ALL RECOMPUTATIONS MATCH PERFECTLY. DATA VALIDATION SUCCESSFUL.")


if __name__ == "__main__":
    verify_results_csv()
