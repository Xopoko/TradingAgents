"""Unattended runs fail closed while interactive runs keep their fallback."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from tradingagents.agents.structured import (
    StructuredOutputError,
    bind_structured,
    invoke_structured,
    invoke_structured_or_freetext,
    strict_structured_outputs,
)


class Decision(BaseModel):
    rating: str


@pytest.mark.parametrize("failure", [NotImplementedError, AttributeError, ValueError])
def test_strict_binding_failure_raises_dedicated_error(failure):
    llm = MagicMock()
    llm.with_structured_output.side_effect = failure("provider failed")

    with strict_structured_outputs(), pytest.raises(StructuredOutputError) as error:
        bind_structured(llm, Decision, "Portfolio Manager")

    assert isinstance(error.value.__cause__, failure)
    llm.invoke.assert_not_called()


def test_strict_binding_rejects_missing_runnable():
    llm = MagicMock()
    llm.with_structured_output.return_value = None
    with strict_structured_outputs(), pytest.raises(StructuredOutputError):
        bind_structured(llm, Decision, "Portfolio Manager")


@pytest.mark.parametrize("result", [None, {"rating": "Buy"}, "Rating: Buy"])
def test_strict_invocation_rejects_unvalidated_response_without_plain_retry(result):
    structured = MagicMock()
    structured.invoke.return_value = result
    plain = MagicMock()

    with strict_structured_outputs(), pytest.raises(StructuredOutputError):
        invoke_structured_or_freetext(
            structured, plain, "prompt", lambda decision: decision.rating, "Trader"
        )

    plain.invoke.assert_not_called()
    structured.invoke.assert_called_once_with("prompt")


def test_strict_invocation_failure_preserves_cause_without_plain_retry():
    structured = MagicMock()
    structured.invoke.side_effect = ValueError("invalid schema response")
    plain = MagicMock()

    with strict_structured_outputs(), pytest.raises(StructuredOutputError) as error:
        invoke_structured_or_freetext(
            structured, plain, "prompt", lambda decision: decision.rating, "Trader"
        )

    assert isinstance(error.value.__cause__, ValueError)
    plain.invoke.assert_not_called()


def test_strict_invocation_rejects_unavailable_binding():
    with strict_structured_outputs(), pytest.raises(StructuredOutputError):
        invoke_structured(None, "prompt", "Trader")


def test_strict_happy_path_returns_validated_model():
    decision = Decision(rating="Buy")
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = decision

    with strict_structured_outputs():
        structured = bind_structured(llm, Decision, "Trader")
        assert invoke_structured(structured, "prompt", "Trader") is decision


def test_nested_context_restores_default_after_exception():
    with pytest.raises(RuntimeError, match="run failed"), strict_structured_outputs():
        with strict_structured_outputs():
            pass
        with pytest.raises(StructuredOutputError):
            invoke_structured(None, "prompt", "Trader")
        raise RuntimeError("run failed")

    assert invoke_structured(None, "prompt", "Trader") is None


def test_strict_setting_does_not_change_an_independent_thread():
    with ThreadPoolExecutor(max_workers=1) as pool, strict_structured_outputs():
        assert pool.submit(invoke_structured, None, "prompt", "Trader").result() is None
        with pytest.raises(StructuredOutputError):
            invoke_structured(None, "prompt", "Trader")


def test_default_fallback_and_unexpected_binding_errors_are_preserved():
    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError("unsupported")
    assert bind_structured(llm, Decision, "Trader") is None
    llm.invoke.return_value.content = "free text"
    assert invoke_structured_or_freetext(
        None, llm, "prompt", lambda decision: decision.rating, "Trader"
    ) == "free text"

    llm.with_structured_output.side_effect = ValueError("invalid configuration")
    with pytest.raises(ValueError, match="invalid configuration"):
        bind_structured(llm, Decision, "Trader")
