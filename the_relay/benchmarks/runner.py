"""Benchmark Runner executing the 5 load scenarios, generating results.csv and Tables 2a & 2b."""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel

from the_relay.config import config
from the_relay.models import (
    ResultsCsvRow,
    RequestStatus,
    SourceType,
)
from the_relay.benchmarks.scenarios import BenchmarkScenarios, ScenarioResult
from the_relay.resilience.dispatcher import ResilientDispatcher
from the_relay.pipeline.core import RelayPipeline
from the_relay.standin.service import StandinModelService, StandinCalibrationConfig
from the_relay.client import RelayModelClient

logger = logging.getLogger(__name__)


class Table2aRow(BaseModel):
    scenario: str
    source: str
    requests_sent: int
    duration_s: float
    offered_rate_per_min: float
    first_token_p50_ms: int
    first_token_p95_ms: int
    first_token_p99_ms: int
    total_time_p50_ms: int
    total_time_p95_ms: int
    total_time_p99_ms: int


class Table2bRow(BaseModel):
    scenario: str
    succeeded_pct: float
    rate_limited_pct: float
    timed_out_or_given_up_pct: float
    retries_per_request: float
    peak_queue_length: int
    tokens_per_answer: int
    cost_per_1000_answers: str


class BenchmarkRunner:
    """Orchestrates all load scenarios, populates results.csv, and computes Tables 2a and 2b."""

    def __init__(
        self,
        dispatcher: Optional[ResilientDispatcher] = None,
        results_dir: Optional[Path] = None,
    ):
        if dispatcher is None:
            # Calibrated stand-in setup for realistic reproducible load runs
            calib = StandinCalibrationConfig(
                base_ttft_mean_ms=180.0,
                base_ttft_std_ms=30.0,
                base_ttft_min_ms=110.0,
                inter_token_delay_ms=12.0,
                load_scale=0.45,
                load_exponent=1.3,
            )
            standin = StandinModelService(calibration=calib)
            client = RelayModelClient(source=SourceType.STAND_IN, standin_service=standin)
            pipeline = RelayPipeline(client=client)
            self.dispatcher = ResilientDispatcher(pipeline=pipeline)
        else:
            self.dispatcher = dispatcher

        self.results_dir = results_dir or config.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.scenarios = BenchmarkScenarios(self.dispatcher)
        self.all_scenario_results: List[ScenarioResult] = []
        self.extra_rows: List[ResultsCsvRow] = []

    def run_all(
        self,
        steady_n: int = 15,
        burst_n: int = 20,
        overload_n: int = 35,
        halved_n: int = 25,
        exhausted_n: int = 15,
        include_retrieval_eval: bool = True,
    ) -> List[ScenarioResult]:
        """Executes all 5 load scenarios and optional retrieval evaluation."""
        results: List[ScenarioResult] = []

        # Check if real Groq key is available for Steady run
        steady_source = SourceType.REAL if self.dispatcher.pipeline.client.groq.is_live_available else SourceType.STAND_IN

        logger.info("Executing Scenario 1: Steady...")
        results.append(self.scenarios.run_steady(n_requests=steady_n, source=steady_source))

        logger.info("Executing Scenario 2: Burst...")
        results.append(self.scenarios.run_burst(burst_size=burst_n, source=SourceType.STAND_IN))

        logger.info("Executing Scenario 3: Sustained Overload...")
        results.append(self.scenarios.run_sustained_overload(n_requests=overload_n, source=SourceType.STAND_IN))

        logger.info("Executing Scenario 4: Per-Minute Limit Halved...")
        results.append(self.scenarios.run_per_minute_limit_halved(n_requests=halved_n, source=SourceType.STAND_IN))

        logger.info("Executing Scenario 5: Daily Budget Exhausted...")
        results.append(self.scenarios.run_daily_budget_exhausted(n_requests=exhausted_n, source=SourceType.STAND_IN))

        self.all_scenario_results = results

        if include_retrieval_eval:
            from the_relay.retrieval.evaluator import RetrievalEvaluator
            evaluator = RetrievalEvaluator(client=self.dispatcher.pipeline.client)
            evaluator.run_all_settings()
            self.extra_rows = evaluator.all_rows

        return results

    def export_results_csv(self, filename: str = "results.csv") -> Path:
        """Writes the mandatory results.csv file with strictly validated schema to results/ and root."""
        out_path = self.results_dir / filename
        root_path = config.base_dir / filename

        all_rows: List[ResultsCsvRow] = []
        for s in self.all_scenario_results:
            all_rows.extend(s.rows)
        all_rows.extend(self.extra_rows)

        fieldnames = [
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

        for p in (out_path, root_path):
            with open(p, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_rows:
                    writer.writerow(row.to_csv_dict())

        return out_path

    def compute_table2a(self) -> List[Table2aRow]:
        """Computes Table 2a (Latency percentiles) directly from scenario runs."""
        table_rows: List[Table2aRow] = []
        for s in self.all_scenario_results:
            ttfts = [r.first_token_ms for r in s.rows if r.first_token_ms > 0]
            totals = [r.total_ms for r in s.rows if r.total_ms > 0]

            ttft_p50 = int(round(np.percentile(ttfts, 50))) if ttfts else 0
            ttft_p95 = int(round(np.percentile(ttfts, 95))) if ttfts else 0
            ttft_p99 = int(round(np.percentile(ttfts, 99))) if ttfts else 0

            tot_p50 = int(round(np.percentile(totals, 50))) if totals else 0
            tot_p95 = int(round(np.percentile(totals, 95))) if totals else 0
            tot_p99 = int(round(np.percentile(totals, 99))) if totals else 0

            table_rows.append(
                Table2aRow(
                    scenario=s.scenario_name,
                    source=s.source.value,
                    requests_sent=s.requests_sent,
                    duration_s=round(s.duration_s, 1),
                    offered_rate_per_min=s.offered_rate_per_min,
                    first_token_p50_ms=ttft_p50,
                    first_token_p95_ms=ttft_p95,
                    first_token_p99_ms=ttft_p99,
                    total_time_p50_ms=tot_p50,
                    total_time_p95_ms=tot_p95,
                    total_time_p99_ms=tot_p99,
                )
            )
        return table_rows

    def compute_table2b(self) -> List[Table2bRow]:
        """Computes Table 2b (Outcomes) directly from scenario runs."""
        table_rows: List[Table2bRow] = []
        for s in self.all_scenario_results:
            n = len(s.rows)
            if n == 0:
                continue

            succeeded = sum(1 for r in s.rows if r.status == RequestStatus.OK)
            rate_limited = sum(1 for r in s.rows if r.status == RequestStatus.RATE_LIMITED)
            timed_out = sum(1 for r in s.rows if r.status in (RequestStatus.TIMEOUT, RequestStatus.ERROR))

            total_retries = sum(r.retries for r in s.rows)
            tokens_out_list = [r.tokens_out for r in s.rows if r.tokens_out > 0]
            avg_tokens = int(round(np.mean(tokens_out_list))) if tokens_out_list else 0

            # Cost per 1,000 answers: Free tier = $0.00 (Equivalent commercial tier = ~$0.32/1k)
            cost_str = "$0.00"

            table_rows.append(
                Table2bRow(
                    scenario=s.scenario_name,
                    succeeded_pct=round((succeeded / n) * 100, 1),
                    rate_limited_pct=round((rate_limited / n) * 100, 1),
                    timed_out_or_given_up_pct=round((timed_out / n) * 100, 1),
                    retries_per_request=round(total_retries / n, 2),
                    peak_queue_length=s.peak_queue_length,
                    tokens_per_answer=avg_tokens,
                    cost_per_1000_answers=cost_str,
                )
            )
        return table_rows

    @staticmethod
    def format_table2a_markdown(rows: List[Table2aRow]) -> str:
        lines = [
            "### Table 2a: Latency, one row per load scenario",
            "",
            "| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r.scenario} | {r.source} | {r.requests_sent} | {r.duration_s} | "
                f"{r.offered_rate_per_min} | {r.first_token_p50_ms} / {r.first_token_p95_ms} / {r.first_token_p99_ms} | "
                f"{r.total_time_p50_ms} / {r.total_time_p95_ms} / {r.total_time_p99_ms} |"
            )
        return "\n".join(lines)

    @staticmethod
    def format_table2b_markdown(rows: List[Table2bRow]) -> str:
        lines = [
            "### Table 2b: Outcomes, same rows",
            "",
            "| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r.scenario} | {r.succeeded_pct}% | {r.rate_limited_pct}% | "
                f"{r.timed_out_or_given_up_pct}% | {r.retries_per_request} | {r.peak_queue_length} | "
                f"{r.tokens_per_answer} | {r.cost_per_1000_answers} |"
            )
        return "\n".join(lines)

