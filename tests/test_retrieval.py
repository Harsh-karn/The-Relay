"""Unit tests for the multi-configuration retrieval engine and Table 3 generation."""

import pytest
from the_relay.retrieval.engine import (
    RelayRetrievalEngine,
    SETTING_1_FINE,
    SETTING_2_BALANCED,
    SETTING_3_COARSE,
)
from the_relay.retrieval.evaluator import RetrievalEvaluator


@pytest.fixture(scope="module")
def engine():
    return RelayRetrievalEngine(setting=SETTING_2_BALANCED)


def test_index_building(engine):
    """Verify chunks are indexed and span all 3 papers."""
    assert len(engine.chunks) > 50
    paper_ids = {c.paper_id for c in engine.chunks}
    assert "1706.03762v7" in paper_ids
    assert "2410.00037v1" in paper_ids
    assert "2507.07955v2" in paper_ids


def test_retrieval_returns_passages_with_citations(engine):
    """Verify queries return top passages with citations."""
    passages = engine.retrieve("What is scaled dot product attention?")
    assert len(passages) == SETTING_2_BALANCED.pieces_fetched
    for p in passages:
        assert p.citation is not None
        assert len(p.citation) > 0
        assert "Page" in p.citation
        assert p.score > 0


def test_degraded_retrieval_mode(engine):
    """Verify degraded mode restricts pieces fetched when quota is low."""
    engine.set_degraded_mode(True)
    passages = engine.retrieve("Moshi Mimi audio codec frame rate")
    assert len(passages) <= 2
    engine.set_degraded_mode(False)


def test_dynamic_setting_switch(engine):
    """Verify switching settings changes chunk counts and fetch behavior."""
    engine.set_setting(SETTING_1_FINE)
    passages = engine.retrieve("warmup steps learning rate schedule")
    assert len(passages) == SETTING_1_FINE.pieces_fetched

    engine.set_setting(SETTING_3_COARSE)
    passages = engine.retrieve("warmup steps learning rate schedule")
    assert len(passages) == SETTING_3_COARSE.pieces_fetched


def test_table3_evaluator_generates_all_three_settings():
    """Verify Table 3 evaluation runs on 15 questions and populates all columns."""
    from the_relay.standin.service import StandinCalibrationConfig, StandinModelService
    from the_relay.client import RelayModelClient
    from the_relay.models import SourceType

    calib = StandinCalibrationConfig(
        base_ttft_mean_ms=2.0,
        base_ttft_std_ms=0.5,
        base_ttft_min_ms=1.0,
        inter_token_delay_ms=0.2,
    )
    service = StandinModelService(calibration=calib)
    client = RelayModelClient(source=SourceType.STAND_IN, standin_service=service)

    evaluator = RetrievalEvaluator(client=client)
    rows = evaluator.run_all_settings()

    assert len(rows) == 3
    for r in rows:
        assert r.questions_asked == 15
        assert r.correct_and_cited_pct + r.wrong_pct + r.correctly_refused_pct == pytest.approx(100.0, abs=0.5)
        assert r.tokens_per_answer > 0
        assert r.first_token_p50_ms > 0
        assert r.total_time_p50_ms >= r.first_token_p50_ms

    md = RetrievalEvaluator.format_table3_markdown(rows)
    assert "### Table 3: Retrieval settings" in md
    assert "200 words, 3 pieces, re-rank no" in md
    assert "500 words, 5 pieces, re-rank yes" in md
    assert "1000 words, 2 pieces, re-rank no" in md

