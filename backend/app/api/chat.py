from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    assistant_id: str
    message: str
    conversation_id: uuid.UUID | None = None


class Citation(BaseModel):
    """Minimal source metadata for a retrieved chunk that grounded a reply.

    Deliberately thin — just enough to identify the document/chunk later.
    Never fabricated: only ever built from chunks the retriever actually
    returned for this principal.
    """

    document_title: str
    chunk_index: int


class ChartSeriesOut(BaseModel):
    name: str
    values: list[float]


class ChartOut(BaseModel):
    """Wire shape of app.orchestration.model_provider.ChartSpec. Present
    only when ChatOrchestrator produced AND validated a chart for this turn
    (see ChatOrchestrator._validate_chart) — never a placeholder/empty chart.
    ``source_chunks`` reuses the same Citation shape text citations use —
    same discipline, a chart's numbers must be traceable to a specific
    retrieved chunk exactly like prose citations are.
    """

    chart_type: Literal["bar", "line", "pie"]
    title: str
    labels: list[str]
    series: list[ChartSeriesOut]
    source_chunks: list[Citation]


class ChatResponse(BaseModel):
    assistant_id: str
    reply: str
    conversation_id: uuid.UUID
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = False
    chart: ChartOut | None = None


class CreateSessionRequest(BaseModel):
    assistant_id: str


class CreateSessionResponse(BaseModel):
    conversation_id: uuid.UUID


class MessageOut(BaseModel):
    role: str
    content: str
    sequence_no: int
    citations: list[Citation] = Field(default_factory=list)
