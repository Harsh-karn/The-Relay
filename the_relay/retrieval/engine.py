"""Multi-configuration RAG Retrieval Engine for The Relay.

Implements parameterizable chunking, vector/lexical indexing, re-ranking,
citation resolution, and degraded retrieval mode when rate limits are near.
"""

import re
import math
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from the_relay.retrieval.ingest import PaperCorpus, PaperDocument


class RetrievalSetting(BaseModel):
    """Configuration for a retrieval setting matching Table 3 requirements."""
    piece_size: int = 500       # Approximate words/tokens per chunk
    pieces_fetched: int = 5     # Number of top pieces returned
    re_rank: bool = True        # Whether re-ranking is enabled
    name: Optional[str] = None

    def display_name(self) -> str:
        rerank_str = "yes" if self.re_rank else "no"
        return f"{self.piece_size} words, {self.pieces_fetched} pieces, re-rank {rerank_str}"


class RetrievedChunk(BaseModel):
    """A retrieved passage with provenance for citations."""
    chunk_id: str
    paper_id: str
    paper_title: str
    page_num: int
    text: str
    score: float
    citation: str


# The 3 canonical benchmark settings for Table 3
SETTING_1_FINE = RetrievalSetting(
    piece_size=200,
    pieces_fetched=3,
    re_rank=False,
    name="Setting 1: Fine chunks (200w, 3 pieces, no re-rank)",
)
SETTING_2_BALANCED = RetrievalSetting(
    piece_size=500,
    pieces_fetched=5,
    re_rank=True,
    name="Setting 2: Balanced chunks (500w, 5 pieces, with re-rank)",
)
SETTING_3_COARSE = RetrievalSetting(
    piece_size=1000,
    pieces_fetched=2,
    re_rank=False,
    name="Setting 3: Coarse chunks (1000w, 2 pieces, no re-rank)",
)


class RelayRetrievalEngine:
    """Indexed retrieval engine operating over the 3 arXiv papers."""

    def __init__(
        self,
        corpus: Optional[PaperCorpus] = None,
        setting: Optional[RetrievalSetting] = None,
    ):
        self.corpus = corpus or PaperCorpus()
        self.setting = setting or SETTING_2_BALANCED
        self.chunks: List[RetrievedChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.degraded_mode: bool = False

        self._build_index()

    def set_setting(self, setting: RetrievalSetting) -> None:
        """Dynamically reconfigures chunking and re-indexes."""
        self.setting = setting
        self._build_index()

    def set_degraded_mode(self, enabled: bool) -> None:
        """Switches to degraded retrieval mode when rate limits are tight."""
        self.degraded_mode = enabled

    def _build_index(self) -> None:
        """Chunks the loaded papers according to piece_size and builds the index."""
        docs = self.corpus.load_corpus()
        self.chunks = []

        chunk_counter = 0
        for paper_id, doc in docs.items():
            author_year = (
                "Vaswani et al. (2017)" if "1706" in paper_id
                else "Defossez et al. (2024)" if "2410" in paper_id
                else "Yu et al. (2025)"
            )

            for page in doc.pages:
                page_words = page.text.split()
                if not page_words:
                    continue

                step = max(50, self.setting.piece_size - 40)  # ~40 words overlap
                for i in range(0, len(page_words), step):
                    chunk_words = page_words[i : i + self.setting.piece_size]
                    if len(chunk_words) < 20 and self.chunks:
                        continue  # Skip trailing micro-fragments

                    chunk_text = " ".join(chunk_words)
                    chunk_counter += 1
                    citation = f"{author_year}, Page {page.page_num}"

                    self.chunks.append(
                        RetrievedChunk(
                            chunk_id=f"{paper_id}_p{page.page_num}_c{chunk_counter}",
                            paper_id=paper_id,
                            paper_title=doc.title,
                            page_num=page.page_num,
                            text=chunk_text,
                            score=0.0,
                            citation=citation,
                        )
                    )

        # Build TF-IDF search index
        corpus_texts = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_df=0.85,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Retrieves most relevant passages for a query."""
        if not self.chunks or self.vectorizer is None:
            return []

        # Determine effective top_k
        k = top_k or self.setting.pieces_fetched

        # When degraded mode is active (due to near-limit quota), reduce fetch volume
        if self.degraded_mode:
            k = min(2, k)

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(similarities)[::-1][: k * 3]  # fetch candidate pool

        candidates: List[RetrievedChunk] = []
        for idx in top_indices:
            score = float(similarities[idx])
            c = self.chunks[idx]
            candidates.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    paper_id=c.paper_id,
                    paper_title=c.paper_title,
                    page_num=c.page_num,
                    text=c.text,
                    score=score,
                    citation=c.citation,
                )
            )

        # Re-ranking stage
        if self.setting.re_rank and not self.degraded_mode:
            candidates = self._rerank(query, candidates)

        return candidates[:k]

    def _rerank(self, query: str, candidates: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Re-scores candidate chunks using query term proximity and title alignment."""
        query_terms = set(re.findall(r"\w+", query.lower()))

        for c in candidates:
            chunk_lower = c.text.lower()
            term_matches = sum(1 for t in query_terms if t in chunk_lower)
            term_density = term_matches / max(1, len(query_terms))

            # Bonus if terms appear in close proximity or paper title relates to query
            title_bonus = 0.15 if any(t in c.paper_title.lower() for t in query_terms) else 0.0

            # Composite re-ranked score
            c.score = float(0.5 * c.score + 0.35 * term_density + 0.15 * title_bonus)

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    def format_context_prompt(self, passages: List[RetrievedChunk]) -> str:
        """Formats retrieved passages into a structured prompt block with citations."""
        if not passages:
            return "No relevant passages found in the library."

        blocks = []
        for i, p in enumerate(passages):
            blocks.append(
                f"[Source {i+1} | {p.citation} | {p.paper_title}]\n{p.text}"
            )
        return "\n\n".join(blocks)

