"""Data structures shared by every stage."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Element:
    """One piece of a parsed document, in reading order."""
    page: int
    kind: str      # "heading" | "text" | "table"
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page_start: int
    page_end: int
    section: str = ""
    content_type: str = "text"     # "text" | "table"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)

    def source_label(self) -> str:
        title = self.metadata.get("title", self.doc_id)
        pages = f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"
        return f"{title}, {pages}" + (f" ({self.section[:60]})" if self.section else "")


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None

    def ref(self) -> dict:
        c = self.chunk
        return {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "page_start": c.page_start, "page_end": c.page_end,
                "section": c.section, "content_type": c.content_type, "label": c.source_label(),
                "score": round(self.score, 5), "dense_rank": self.dense_rank, "bm25_rank": self.bm25_rank,
                "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 4)}
