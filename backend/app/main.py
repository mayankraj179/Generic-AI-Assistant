from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.api.chat import (
    ChartOut,
    ChartSeriesOut,
    ChatRequest,
    ChatResponse,
    Citation,
    CreateSessionRequest,
    CreateSessionResponse,
    MessageOut,
)
from app.api.ingest import AdminIngestRequest, AdminIngestResponse
from app.auth.dependencies import require_permission
from app.auth.provider import JwtAuthProvider
from app.config.loader import load_all_assistant_configs
from app.config.settings import Settings
from app.core.principal import PrincipalContext
from app.ingestion.pipeline import ParseError
from app.observability.context import new_request_id, request_id_var
from app.observability.logging_setup import configure_logging
from app.observability.provider_errors import find_failure, log_provider_failure
from app.observability.tracing import configure_tracing, start_span
from app.observability.usage import usage_summary
from app.orchestration.chat_service import (
    ChatOrchestrator,
    ChatStreamChart,
    ChatStreamDelta,
    ChatStreamDone,
    ChatStreamError,
    ChatStreamEvent,
    build_default_chat_orchestrator,
)
from app.orchestration.errors import (
    AssistantAccessDeniedError,
    InputGuardrailError,
    RetrievalUnavailableError,
)
from app.orchestration.model_provider import ChartSpec, ModelProviderError
from app.services.conversation_store import ConversationNotFoundError, ConversationStore
from app.services.embedding import EmbeddingProviderError
from app.services.embedding_provider_factory import get_embedding_provider
from app.services.ingestion_service import IngestionService

configure_logging()
configure_tracing()
logger = logging.getLogger(__name__)

settings = Settings()

app = FastAPI(title="Generic AI Assistant Framework", version="0.1.0")

_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Assigns the request id every log line and audit row carries (a
    well-formed incoming X-Request-ID is kept so a caller can correlate), and
    opens the request's root span. google-adk's own spans nest under it."""
    incoming = request.headers.get("X-Request-ID", "")
    request_id = incoming if _REQUEST_ID_RE.fullmatch(incoming) else new_request_id()
    request_id_var.set(request_id)
    with start_span(
        f"{request.method} {request.url.path}",
        **{
            "http.method": request.method,
            "http.target": request.url.path,
            "request_id": request_id,
        },
    ) as span:
        response = await call_next(request)
        span.set_attribute("http.status_code", response.status_code)
    response.headers["X-Request-ID"] = request_id
    return response

app.state.settings = settings
app.state.auth_provider = JwtAuthProvider(settings=settings)
app.state.chat_orchestrator = build_default_chat_orchestrator(settings=settings)
app.state.conversation_store = ConversationStore()
app.state.ingestion_service = IngestionService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def get_chat_orchestrator(request: Request) -> ChatOrchestrator:
    orchestrator = getattr(request.app.state, "chat_orchestrator", None)
    if orchestrator is None:
        orchestrator = build_default_chat_orchestrator(settings=request.app.state.settings)
        request.app.state.chat_orchestrator = orchestrator
    return orchestrator


def get_conversation_store(request: Request) -> ConversationStore:
    store = getattr(request.app.state, "conversation_store", None)
    if store is None:
        store = ConversationStore()
        request.app.state.conversation_store = store
    return store


def get_ingestion_service(request: Request) -> IngestionService:
    service = getattr(request.app.state, "ingestion_service", None)
    if service is None:
        service = IngestionService()
        request.app.state.ingestion_service = service
    return service


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/assistants")
async def list_assistants(
    principal: Annotated[
        PrincipalContext,
        Depends(require_permission("assistant:use")),
    ],
) -> list[dict[str, str]]:
    del principal
    configs = load_all_assistant_configs(CONFIGS_DIR)
    return [
        {"assistant_id": c.assistant_id, "display_name": c.display_name}
        for c in configs.values()
    ]


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    principal: Annotated[
        PrincipalContext,
        Depends(require_permission("assistant:use")),
    ],
    orchestrator: Annotated[ChatOrchestrator, Depends(get_chat_orchestrator)],
) -> ChatResponse:
    configs = load_all_assistant_configs(CONFIGS_DIR)
    config = configs.get(request.assistant_id)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown assistant_id '{request.assistant_id}'",
        )

    try:
        result = await orchestrator.handle(
            principal=principal,
            config=config,
            message=request.message,
            conversation_id=request.conversation_id,
        )
    except InputGuardrailError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except AssistantAccessDeniedError:
        raise HTTPException(
            status_code=403,
            detail="principal is not permitted to use this assistant",
        ) from None
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="conversation not found",
        ) from None
    except RetrievalUnavailableError as exc:
        # Already logged once, classified, by ChatOrchestrator.
        raise HTTPException(
            status_code=503,
            detail="retrieval is temporarily unavailable",
        ) from exc
    except ModelProviderError as exc:
        # Already logged once, classified, by ChatOrchestrator.
        raise HTTPException(
            status_code=503,
            detail="the assistant model is temporarily unavailable",
        ) from exc

    return ChatResponse(
        assistant_id=config.assistant_id,
        reply=result.text,
        conversation_id=result.conversation_id,
        citations=[
            Citation(document_title=c.document_title, chunk_index=c.chunk_index)
            for c in result.citations
        ],
        grounded=result.grounded,
        chart=_to_chart_out(result.chart),
    )


def _to_chart_out(chart: ChartSpec | None) -> ChartOut | None:
    if chart is None:
        return None
    return ChartOut(
        chart_type=chart.chart_type,
        title=chart.title,
        labels=chart.labels,
        series=[ChartSeriesOut(name=s.name, values=s.values) for s in chart.series],
        source_chunks=[
            Citation(document_title=sc.document_title, chunk_index=sc.chunk_index)
            for sc in chart.source_chunks
        ],
    )


def _encode_sse_event(event: ChatStreamEvent) -> str:
    """Formats one stream event as an SSE ``data:`` line. Payload shapes are
    intentionally the smallest possible JSON — no field beyond what the
    frontend actually consumes."""
    if isinstance(event, ChatStreamDelta):
        payload = {"type": "delta", "text": event.text}
    elif isinstance(event, ChatStreamChart):
        chart_out = _to_chart_out(event.chart)
        assert chart_out is not None  # ChatStreamChart is only ever constructed with a real chart
        payload = {"type": "chart", "chart": chart_out.model_dump()}
    elif isinstance(event, ChatStreamDone):
        payload = {
            "type": "done",
            "conversation_id": str(event.conversation_id),
            "citations": [
                {"document_title": c.document_title, "chunk_index": c.chunk_index}
                for c in event.citations
            ],
            "grounded": event.grounded,
        }
    elif isinstance(event, ChatStreamError):
        payload = {"type": "error", "detail": event.detail}
    else:  # pragma: no cover - exhaustive over ChatStreamEvent's members today
        raise ModelProviderError(f"unrecognized stream event type: {type(event)!r}")
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    principal: Annotated[
        PrincipalContext,
        Depends(require_permission("assistant:use")),
    ],
    orchestrator: Annotated[ChatOrchestrator, Depends(get_chat_orchestrator)],
) -> StreamingResponse:
    configs = load_all_assistant_configs(CONFIGS_DIR)
    config = configs.get(request.assistant_id)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown assistant_id '{request.assistant_id}'",
        )

    stream = orchestrator.handle_stream(
        principal=principal,
        config=config,
        message=request.message,
        conversation_id=request.conversation_id,
    )

    # Pull the first event eagerly, before returning the StreamingResponse.
    # handle_stream() raises AssistantAccessDeniedError / ConversationNotFoundError
    # / RetrievalUnavailableError from its setup phase before yielding
    # anything, so doing this here — rather than inside the response body —
    # is what lets this route return the same real HTTP status codes /chat
    # does for those failures. Once the response has started, headers are
    # already sent and a status code can no longer change; failures from
    # that point on (ModelProviderError) are handled entirely inside
    # handle_stream() as an in-band {"type": "error"} event instead.
    try:
        first_event = await anext(stream)
    except StopAsyncIteration:
        raise HTTPException(
            status_code=503,
            detail="the assistant model is temporarily unavailable",
        ) from None
    except InputGuardrailError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except AssistantAccessDeniedError:
        raise HTTPException(
            status_code=403,
            detail="principal is not permitted to use this assistant",
        ) from None
    except ConversationNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="conversation not found",
        ) from None
    except RetrievalUnavailableError as exc:
        # Already logged once, classified, by ChatOrchestrator.
        raise HTTPException(
            status_code=503,
            detail="retrieval is temporarily unavailable",
        ) from exc

    async def event_source() -> AsyncIterator[str]:
        yield _encode_sse_event(first_event)
        async for event in stream:
            yield _encode_sse_event(event)

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    request: CreateSessionRequest,
    principal: Annotated[
        PrincipalContext,
        Depends(require_permission("assistant:use")),
    ],
    conversation_store: Annotated[ConversationStore, Depends(get_conversation_store)],
) -> CreateSessionResponse:
    configs = load_all_assistant_configs(CONFIGS_DIR)
    if request.assistant_id not in configs:
        raise HTTPException(
            status_code=404,
            detail=f"unknown assistant_id '{request.assistant_id}'",
        )

    conversation = await conversation_store.create_conversation(
        principal=principal, assistant_id=request.assistant_id
    )
    return CreateSessionResponse(conversation_id=conversation.id)


@app.get("/sessions/{conversation_id}/messages", response_model=list[MessageOut])
async def get_session_messages(
    conversation_id: uuid.UUID,
    principal: Annotated[
        PrincipalContext,
        Depends(require_permission("assistant:use")),
    ],
    conversation_store: Annotated[ConversationStore, Depends(get_conversation_store)],
) -> list[MessageOut]:
    try:
        messages = await conversation_store.list_messages(
            conversation_id=conversation_id, principal=principal
        )
    except ConversationNotFoundError:
        # Never distinguish "doesn't exist" from "exists but isn't yours" —
        # both return 404, so a caller can't fingerprint other principals'
        # conversation IDs.
        raise HTTPException(
            status_code=404,
            detail="conversation not found",
        ) from None

    return [
        MessageOut(
            role=message.role,
            content=message.content,
            sequence_no=message.sequence_no,
            citations=[
                Citation(document_title=c.document_title, chunk_index=c.chunk_index)
                for c in message.citations
            ],
        )
        for message in messages
    ]


@app.post("/admin/ingest", response_model=AdminIngestResponse)
async def admin_ingest(
    request: AdminIngestRequest,
    principal: Annotated[
        PrincipalContext,
        Depends(require_permission("admin:ingest")),
    ],
    ingestion_service: Annotated[IngestionService, Depends(get_ingestion_service)],
) -> AdminIngestResponse:
    del principal
    configs = load_all_assistant_configs(CONFIGS_DIR)
    config = configs.get(request.assistant_id)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown assistant_id '{request.assistant_id}'",
        )

    access_labels = frozenset(request.access_labels) if request.access_labels else None
    embedder = get_embedding_provider(config=config, settings=settings)

    try:
        chunks = await ingestion_service.ingest_file(
            source_uri=request.source_path,
            tenant_id=request.tenant_id,
            assistant_id=request.assistant_id,
            access_labels=access_labels,
            embedder=embedder,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except ParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except EmbeddingProviderError as exc:
        failure = find_failure(exc)
        if failure is not None:
            log_provider_failure(logger, failure, exc)
        else:
            logger.error("embedding failed while ingesting '%s': %s", request.source_path, exc)
        raise HTTPException(
            status_code=503,
            detail="the embedding provider is temporarily unavailable",
        ) from exc

    return AdminIngestResponse(source_uri=request.source_path, chunks_ingested=len(chunks))


@app.get("/admin/usage")
async def admin_usage(
    principal: Annotated[PrincipalContext, Depends(require_permission("admin:ingest"))],
) -> dict:
    """Rough provider-quota usage in each provider's current window, from the
    turn audit (see app/observability/usage.py for what it does and doesn't
    count). Uses admin:ingest, the only admin permission the policy defines."""
    del principal
    lines = await usage_summary()
    return {
        "note": "chat turns only (ingestion/scripts not audited): a lower bound",
        "usage": [
            {
                "provider": u.provider,
                "kind": u.kind,
                "model": u.model,
                "calls": u.calls,
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "limit": u.limit,
                "limit_source": u.limit_source,
                "window_start": u.window_start.isoformat(),
                "last_quota_error": u.last_quota_error,
            }
            for u in lines
        ],
    }
