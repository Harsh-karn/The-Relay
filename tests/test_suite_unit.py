"""Tests for the 15 benchmark questions and grading logic."""

import pytest
from the_relay.retrieval.test_suite import (
    TestSuiteManager,
    BenchmarkQuestion,
    NAIVE_RETRIEVER_PITFALLS,
)
from the_relay.models import AnswerGrade


def test_test_suite_count_and_categories():
    """Verify exactly 15 questions exist with proper distribution across categories."""
    mgr = TestSuiteManager()
    assert len(mgr.questions) == 15

    grounded = [q for q in mgr.questions if q.category == "grounded"]
    traps = [q for q in mgr.questions if q.category == "adversarial_trap"]
    unanswerable = [q for q in mgr.questions if q.category == "unanswerable"]

    assert len(grounded) == 8
    assert len(traps) == 3
    assert len(unanswerable) == 4

    for q in unanswerable:
        assert q.must_refuse is True


def test_naive_retriever_pitfalls_complete():
    """Verify 3 distinct pitfalls are defined with explanation and example question."""
    assert len(NAIVE_RETRIEVER_PITFALLS) == 3
    for p in NAIVE_RETRIEVER_PITFALLS:
        assert "pitfall" in p
        assert len(p["explanation"]) > 30
        assert "example_question" in p


def test_grading_correctly_refused():
    """Verify correct refusal on out-of-domain question."""
    mgr = TestSuiteManager()
    q = next(q for q in mgr.questions if q.must_refuse)
    ans = "The provided papers do not say anything about ViT patch projection."
    grade = mgr.grade_answer(q, ans, [])
    assert grade == AnswerGrade.REFUSED_CORRECTLY


def test_grading_refused_wrongly():
    """Verify when an unanswerable question produces a hallucinated answer without refusal."""
    mgr = TestSuiteManager()
    q = next(q for q in mgr.questions if q.must_refuse)
    ans = "ViT projects 224x224 into 16x16 linearly with 768 dimensions."
    grade = mgr.grade_answer(q, ans, [])
    assert grade == AnswerGrade.REFUSED_WRONGLY


def test_grading_correct_grounded_answer():
    """Verify grounded answer grading with matching keywords and citations."""
    mgr = TestSuiteManager()
    q = next(q for q in mgr.questions if q.id == "q01_vaswani_dimensions")
    ans = "In Section 3.2.2, Vaswani et al. (2017) use h=8 parallel heads with d_k = d_v = 64 and d_model = 512."
    grade = mgr.grade_answer(q, ans, ["Vaswani et al. (2017), Section 3.2.2"])
    assert grade == AnswerGrade.CORRECT

