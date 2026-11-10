"""Unified Model Client abstracting Real Groq and Stand-in services."""

from typing import List, Optional, Iterator
from the_relay.models import (
    ChatMessage,
    CompletionResponse,
    StreamChunk,
    SourceType,
    RequestStatus,
)
from the_relay.profiler.groq_profiler import GroqProfiler
from the_relay.standin.service import StandinModelService


class RelayModelClient:
    """Unified client routing requests to either Real Groq or the Stand-in model service."""

    def __init__(
        self,
        source: SourceType = SourceType.STAND_IN,
        standin_service: Optional[StandinModelService] = None,
        groq_profiler: Optional[GroqProfiler] = None,
    ):
        self.source = source
        self.standin = standin_service or StandinModelService()
        self.groq = groq_profiler or GroqProfiler()

    def stream(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 256,
        synthetic_answer: Optional[str] = None,
    ) -> Iterator[StreamChunk]:
        """Streams tokens from the active backend."""
        if self.source == SourceType.STAND_IN:
            yield from self.standin.stream_completion(
                messages=messages,
                max_tokens=max_tokens,
                synthetic_answer=synthetic_answer,
            )
        else:
            # For real Groq streaming
            prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
            resp = self.groq.send_single_streaming_request(prompt=prompt)
            # Yield as single or split chunks
            words = resp.text.split(" ")
            first = True
            for idx, w in enumerate(words):
                yield StreamChunk(
                    delta=w + (" " if idx < len(words) - 1 else ""),
                    index=idx,
                    elapsed_ms=resp.first_token_ms if first else resp.total_ms,
                    is_first_token=first,
                    finish_reason="stop" if idx == len(words) - 1 else None,
                )
                first = False

    def complete(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 256,
        synthetic_answer: Optional[str] = None,
    ) -> CompletionResponse:
        """Executes full completion through active backend."""
        if self.source == SourceType.STAND_IN:
            return self.standin.generate_complete(
                messages=messages,
                max_tokens=max_tokens,
                synthetic_answer=synthetic_answer,
            )
        else:
            prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
            return self.groq.send_single_streaming_request(prompt=prompt)
