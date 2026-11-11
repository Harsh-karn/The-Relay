"""Unit tests for the 4-step resilient pipeline."""

import pytest
from the_relay.pipeline.core import RelayPipeline
from the_relay.pipeline.models import PipelineBranch
from the_relay.models import RequestStatus, SourceType
from the_relay.standin.service import StandinModelService, StandinCalibrationConfig
from the_relay.client import RelayModelClient


@pytest.fixture
def fast_pipeline():
    calib = StandinCalibrationConfig(
        base_ttft_mean_ms=2.0,
        base_ttft_std_ms=0.5,
        base_ttft_min_ms=1.0,
        inter_token_delay_ms=0.2,
    )
    service = StandinModelService(calibration=calib)
    client = RelayModelClient(source=SourceType.STAND_IN, standin_service=service)
    return RelayPipeline(client=client)


def test_four_steps_executed_in_order(fast_pipeline):
    """Verify all 4 steps execute in proper numerical order."""
    res = fast_pipeline.process_query("What are the attention head dimensions d_k and d_v in Vaswani?")
    assert res.status == RequestStatus.OK
    assert len(res.steps_executed) >= 4

    step_nums = [s.step_number for s in res.steps_executed]
    assert 1 in step_nums
    assert 2 in step_nums
    assert 3 in step_nums
    assert 4 in step_nums


def test_tool_call_recorded(fast_pipeline):
    """Verify tool call to retrieve_passages is logged with arguments and duration."""
    res = fast_pipeline.process_query("How many codebooks does Mimi use?")
    assert len(res.tool_calls) >= 1
    tc = res.tool_calls[0]
    assert tc.tool_name == "retrieve_passages"
    assert "query" in tc.arguments
    assert tc.duration_ms > 0
    assert "Retrieved" in tc.output_summary


def test_out_of_domain_branches_to_refusal(fast_pipeline):
    """Verify unanswerable/out-of-domain query branches to refusal without hallucination."""
    res = fast_pipeline.process_query("How does the Vision Transformer project 224x224 images into patches?")
    assert res.branch_taken == PipelineBranch.REFUSAL
    assert "papers do not say" in res.answer.lower()
    assert len(res.citations) == 0


def test_grounded_query_branches_to_answer_with_citations(fast_pipeline):
    """Verify in-domain question branches to grounded answer with citations."""
    res = fast_pipeline.process_query("What BLEU score was achieved on WMT 2014 English-to-French?")
    assert res.branch_taken == PipelineBranch.GROUNDED_ANSWER
    assert len(res.citations) > 0
    assert any("Vaswani et al. (2017)" in c for c in res.citations)
    assert "(" in res.answer and ")" in res.answer


def test_refinement_loop_on_ambiguous_query(fast_pipeline):
    """Verify query refinement loop triggers when ambiguous term is encountered."""
    # "chunking" alone without context triggers refinement loop in Step 3
    res = fast_pipeline.process_query("What are the chunking boundaries?")
    assert res.loop_count >= 1
    # Check that Step 2 was executed more than once
    step2_runs = [s for s in res.steps_executed if s.step_number == 2]
    assert len(step2_runs) >= 2


def test_latency_breakdown_complete(fast_pipeline):
    """Verify waiting, working, and first_token latency are split accurately."""
    res = fast_pipeline.process_query(
        "Explain warmup steps in learning rate schedule",
        queue_wait_ms=45.0,
    )
    assert res.latency.waiting_ms == 45.0
    assert res.latency.working_ms > 0
    assert res.latency.first_token_ms >= 45.0
    assert res.latency.total_ms >= res.latency.first_token_ms

