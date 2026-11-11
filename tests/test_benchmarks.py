"""Unit tests for the benchmark load runner and Table 2a/2b generation."""

import pytest
import csv
from pathlib import Path

from the_relay.benchmarks.runner import BenchmarkRunner
from the_relay.models import RequestStatus, SourceType
from the_relay.resilience.dispatcher import ResilientDispatcher
from the_relay.pipeline.core import RelayPipeline
from the_relay.standin.service import StandinModelService, StandinCalibrationConfig
from the_relay.client import RelayModelClient


@pytest.fixture
def fast_runner(tmp_path):
    """Configures a high-speed runner for fast unit testing."""
    calib = StandinCalibrationConfig(
        base_ttft_mean_ms=1.5,
        base_ttft_std_ms=0.3,
        base_ttft_min_ms=1.0,
        inter_token_delay_ms=0.1,
    )
    service = StandinModelService(calibration=calib)
    client = RelayModelClient(source=SourceType.STAND_IN, standin_service=service)
    pipeline = RelayPipeline(client=client)
    dispatcher = ResilientDispatcher(pipeline=pipeline)
    return BenchmarkRunner(dispatcher=dispatcher, results_dir=tmp_path)


def test_runner_executes_all_five_scenarios(fast_runner):
    """Verify all 5 scenarios execute and return valid ScenarioResult objects."""
    results = fast_runner.run_all(
        steady_n=4,
        burst_n=6,
        overload_n=10,
        halved_n=6,
        exhausted_n=4,
        include_retrieval_eval=False,
    )
    assert len(results) == 5
    names = [r.scenario_name for r in results]
    assert "steady" in names
    assert "burst" in names
    assert "sustained_overload" in names
    assert "per_minute_limit_halved" in names
    assert "daily_budget_exhausted" in names


def test_export_results_csv_schema(fast_runner, tmp_path):
    """Verify results.csv conforms to the mandatory assignment schema."""
    fast_runner.run_all(
        steady_n=2,
        burst_n=3,
        overload_n=4,
        halved_n=3,
        exhausted_n=2,
        include_retrieval_eval=False,
    )
    csv_path = fast_runner.export_results_csv()
    assert csv_path.exists()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        expected_fields = [
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
        assert fieldnames == expected_fields
        rows = list(reader)
        assert len(rows) == (2 + 3 + 4 + 3 + 2)
        for r in rows:
            assert r["status"] in ("ok", "rate_limited", "timeout", "error")
            assert r["source"] in ("real", "stand-in")
            assert r["answer_grade"] in ("correct", "wrong", "refused_correctly", "refused_wrongly")


def test_compute_tables_2a_and_2b(fast_runner):
    """Verify Table 2a and 2b rows are populated correctly."""
    fast_runner.run_all(
        steady_n=3,
        burst_n=4,
        overload_n=6,
        halved_n=4,
        exhausted_n=3,
        include_retrieval_eval=False,
    )
    t2a = fast_runner.compute_table2a()
    t2b = fast_runner.compute_table2b()

    assert len(t2a) == 5
    assert len(t2b) == 5

    # Check daily budget exhausted in Table 2b has 100% rate limited
    exhausted_row = next(r for r in t2b if r.scenario == "daily_budget_exhausted")
    assert exhausted_row.rate_limited_pct == 100.0
    assert exhausted_row.succeeded_pct == 0.0

    # Markdown formatters
    md2a = BenchmarkRunner.format_table2a_markdown(t2a)
    md2b = BenchmarkRunner.format_table2b_markdown(t2b)
    assert "### Table 2a: Latency" in md2a
    assert "### Table 2b: Outcomes" in md2b

