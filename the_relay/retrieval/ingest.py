"""Corpus ingestion for arXiv research papers.

Fetches the exact paper versions:
- Attention Is All You Need, arXiv 1706.03762, version 7
- Moshi: a speech-text foundation model for real-time dialogue, arXiv 2410.00037, version 1
- Dynamic Chunking for End-to-End Hierarchical Sequence Modeling, arXiv 2507.07955, version 2
Extracts page-by-page text and structural provenance metadata.
"""

import os
import json
import logging
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel
import fitz  # PyMuPDF

from the_relay.config import config

logger = logging.getLogger(__name__)

PAPERS_SPEC = [
    {
        "id": "1706.03762v7",
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "version": 7,
        "url": "https://arxiv.org/pdf/1706.03762v7.pdf",
    },
    {
        "id": "2410.00037v1",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "arxiv_id": "2410.00037",
        "version": 1,
        "url": "https://arxiv.org/pdf/2410.00037v1.pdf",
    },
    {
        "id": "2507.07955v2",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "arxiv_id": "2507.07955",
        "version": 2,
        "url": "https://arxiv.org/pdf/2507.07955v2.pdf",
    },
]


class PaperPage(BaseModel):
    page_num: int
    text: str


class PaperDocument(BaseModel):
    paper_id: str
    arxiv_id: str
    version: int
    title: str
    total_pages: int
    pages: List[PaperPage]

    def full_text(self) -> str:
        return "\n\n".join(f"[Page {p.page_num}]\n{p.text}" for p in self.pages)


class PaperCorpus:
    """Manages downloading, parsing, and caching the 3 arXiv research papers."""

    def __init__(self, papers_dir: Optional[Path] = None):
        self.papers_dir = papers_dir or (config.data_dir / "papers")
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.documents: Dict[str, PaperDocument] = {}

    def fetch_all(self, force_download: bool = False) -> None:
        """Downloads all 3 target PDFs if not already cached locally."""
        for spec in PAPERS_SPEC:
            pdf_path = self.papers_dir / f"{spec['id']}.pdf"
            if not pdf_path.exists() or force_download:
                logger.info(f"Downloading {spec['id']} from {spec['url']}...")
                req = urllib.request.Request(
                    spec["url"],
                    headers={"User-Agent": "TheRelay-Research-Agent/1.0"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp, open(pdf_path, "wb") as f:
                    f.write(resp.read())
                logger.info(f"Saved {pdf_path} ({pdf_path.stat().st_size} bytes)")

    def parse_all(self) -> Dict[str, PaperDocument]:
        """Extracts text from downloaded PDFs, saving page-by-page JSON cache."""
        for spec in PAPERS_SPEC:
            paper_id = spec["id"]
            json_path = self.papers_dir / f"{paper_id}.json"
            pdf_path = self.papers_dir / f"{paper_id}.pdf"

            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.documents[paper_id] = PaperDocument(**data)
                continue

            if not pdf_path.exists():
                raise FileNotFoundError(
                    f"PDF {pdf_path} not found. Run fetch_all() with network access first."
                )

            # Extract via PyMuPDF
            doc = fitz.open(pdf_path)
            extracted_pages: List[PaperPage] = []
            for i, page in enumerate(doc):
                text = page.get_text("text").strip()
                extracted_pages.append(PaperPage(page_num=i + 1, text=text))

            doc_model = PaperDocument(
                paper_id=paper_id,
                arxiv_id=spec["arxiv_id"],
                version=spec["version"],
                title=spec["title"],
                total_pages=len(extracted_pages),
                pages=extracted_pages,
            )

            with open(json_path, "w", encoding="utf-8") as f:
                f.write(doc_model.model_dump_json(indent=2))

            self.documents[paper_id] = doc_model

        return self.documents

    def load_corpus(self) -> Dict[str, PaperDocument]:
        """Loads all documents from cache or parses if needed."""
        if not self.documents:
            self.parse_all()
        return self.documents
