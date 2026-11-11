"""Retrieval package for The Relay."""

from the_relay.retrieval.ingest import PaperCorpus, PaperDocument, PaperPage
from the_relay.retrieval.engine import (
    RelayRetrievalEngine,
    RetrievalSetting,
    RetrievedChunk,
    SETTING_1_FINE,
    SETTING_2_BALANCED,
    SETTING_3_COARSE,
)
from the_relay.retrieval.test_suite import (
    TestSuiteManager,
    BenchmarkQuestion,
    NAIVE_RETRIEVER_PITFALLS,
)
from the_relay.retrieval.evaluator import RetrievalEvaluator, Table3Row

__all__ = [
    "PaperCorpus",
    "PaperDocument",
    "PaperPage",
    "RelayRetrievalEngine",
    "RetrievalSetting",
    "RetrievedChunk",
    "SETTING_1_FINE",
    "SETTING_2_BALANCED",
    "SETTING_3_COARSE",
    "TestSuiteManager",
    "BenchmarkQuestion",
    "NAIVE_RETRIEVER_PITFALLS",
    "RetrievalEvaluator",
    "Table3Row",
]
