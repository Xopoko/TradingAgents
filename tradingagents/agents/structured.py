"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Portfolio Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call itself fails for any reason
   (malformed JSON from a weak model, transient provider issue), fall
   back to a plain ``llm.invoke`` so the pipeline never blocks.

Centralising the pattern here keeps the agent factories small and ensures
all three agents log the same warnings when fallback fires.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_strict_structured_outputs: ContextVar[bool] = ContextVar(
    "strict_structured_outputs", default=False
)


class StructuredOutputError(RuntimeError):
    """A required structured response could not be bound or validated."""


@contextmanager
def strict_structured_outputs() -> Iterator[None]:
    """Require typed output without free-text fallback within this context.

    Wrap both agent construction and execution. The setting is local to the
    current context and is restored even when a run fails.
    """
    token = _strict_structured_outputs.set(True)
    try:
        yield
    finally:
        _strict_structured_outputs.reset(token)

# Schema-only structured output binds exactly one tool (the schema itself), so a
# model that reaches for a search tool emits an unknown tool call and the whole
# structured attempt is discarded for a free-text retry. Agents on this path
# state the constraint explicitly rather than relying on the binding alone
# (#1130).
NO_EXTERNAL_TOOLS = (
    "Use only the evidence provided in this prompt. Do not call external tools "
    "or search the web; if something is missing, say so explicitly."
)


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Any | None:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        structured_llm = llm.with_structured_output(schema)
    except Exception as exc:
        if _strict_structured_outputs.get():
            raise StructuredOutputError(
                f"{agent_name}: structured-output binding failed"
            ) from exc
        if not isinstance(exc, (NotImplementedError, AttributeError)):
            raise
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None
    if structured_llm is None and _strict_structured_outputs.get():
        raise StructuredOutputError(
            f"{agent_name}: structured-output binding returned no runnable"
        )
    return structured_llm


def invoke_structured(structured_llm: Any | None, prompt: Any, agent_name: str) -> T | None:
    """Run the structured call; ``None`` when there is none or it fails.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape), so a caller can forward the same value to its free-text fallback.
    """
    if structured_llm is None:
        if _strict_structured_outputs.get():
            raise StructuredOutputError(
                f"{agent_name}: structured output is unavailable"
            )
        return None
    try:
        result = structured_llm.invoke(prompt)
        if result is None:
            # A thinking model can answer in plain text instead of calling
            # the tool, leaving the parser with nothing to return. Treat it
            # as a structured miss and fall back, with a clear reason.
            raise ValueError("structured output returned no parsed result")
        if _strict_structured_outputs.get() and not isinstance(result, BaseModel):
            raise TypeError("structured output did not return a validated model")
        return result
    except Exception as exc:
        if _strict_structured_outputs.get():
            raise StructuredOutputError(
                f"{agent_name}: structured-output invocation failed"
            ) from exc
        logger.warning(
            "%s: structured-output invocation failed (%s); retrying once as free text",
            agent_name, exc,
        )
        return None


def invoke_structured_or_freetext(
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> str:
    """Run the structured call and render to markdown; fall back to free-text on any failure."""
    result = invoke_structured(structured_llm, prompt, agent_name)
    if result is not None:
        return render(result)
    return plain_llm.invoke(prompt).content
