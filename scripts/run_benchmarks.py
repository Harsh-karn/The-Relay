"""Script to run the full benchmark suite across all 5 scenarios and generate results.csv, Table 2a, and Table 2b."""

import logging
from the_relay.benchmarks.runner import BenchmarkRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    print("=" * 60)
    print("Executing The Relay Benchmark Suite (All 5 Scenarios)")
    print("=" * 60)

    runner = BenchmarkRunner()
    results = runner.run_all(
        steady_n=15,
        burst_n=20,
        overload_n=35,
        halved_n=25,
        exhausted_n=15,
    )

    csv_path = runner.export_results_csv("results.csv")
    print(f"\nSaved raw results to {csv_path} ({csv_path.stat().st_size} bytes)")

    t2a_rows = runner.compute_table2a()
    t2b_rows = runner.compute_table2b()

    md2a = runner.format_table2a_markdown(t2a_rows)
    md2b = runner.format_table2b_markdown(t2b_rows)

    with open(runner.results_dir / "table2a.md", "w", encoding="utf-8") as f:
        f.write(md2a)
    with open(runner.results_dir / "table2b.md", "w", encoding="utf-8") as f:
        f.write(md2b)

    print("\n" + md2a)
    print("\n" + md2b)
    print("\nBenchmark generation completed successfully.")

if __name__ == "__main__":
    main()

