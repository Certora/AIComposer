"""
Tests for IndexedTool infrastructure: cache hits, misses, and re-caching.

Requires Docker (uses testcontainers for a throwaway Postgres instance).
"""
import pytest
import pytest_asyncio

from typing import override
from pydantic import Field

from langgraph.graph import MessagesState

from composer.spec.agent_index import AgentIndex, IndexedTool

from graphcore.tests.conftest import Scenario, tool_call_raw, ToolCallDict

from .conftest import needs_postgres

pytestmark = [pytest.mark.asyncio, needs_postgres]


# ---------------------------------------------------------------------------
# Test tool
# ---------------------------------------------------------------------------

answer_question_calls: list[str] = []

_ASK = "ask_question"


class AskQuestion(IndexedTool[AgentIndex]):
    """Ask a question about the codebase."""
    question: str = Field(description="The question to ask")

    @override
    def get_question(self) -> str:
        return self.question

    @override
    async def answer_question(self, context: list[str]) -> str:
        answer_question_calls.append(self.question)
        return f"FRESH ANSWER (context had {len(context)} prior matches)"


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

SEED_DATA = [
    {"question": "What color is the sky?", "answer": "Blue on a clear day."},
    {"question": "How does TCP handle congestion?", "answer": "Slow start, congestion avoidance, fast retransmit."},
    {"question": "What is the boiling point of water?", "answer": "100 degrees Celsius at sea level."},
    {"question": "Explain the Pythagorean theorem.", "answer": "a^2 + b^2 = c^2 for right triangles."},
    {
        "question": "What are the state variables of the Vault contract?",
        "answer": (
            "The Vault contract tracks:\n"
            "- `mapping(address => uint256) balances`\n"
            "- `uint256 totalDeposits`\n"
            "- `address owner`\n"
            "- `bool paused`"
        ),
    },
]


# ---------------------------------------------------------------------------
# Tool call constructors
# ---------------------------------------------------------------------------


def _ask(question: str) -> ToolCallDict:
    return tool_call_raw(_ASK, question=question)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def ask_scenario(index: AgentIndex):
    for doc in SEED_DATA:
        await index.aput(**doc)
    answer_question_calls.clear()
    tool = AskQuestion.bind(index).as_tool(_ASK)
    return Scenario(MessagesState, tool).init()


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


def _response(st: MessagesState) -> str:
    return Scenario.last_single_tool(_ASK, st)


def _call_count(_: MessagesState) -> int:
    return len(answer_question_calls)


def _response_and_calls(st: MessagesState) -> tuple[str, int]:
    return _response(st), _call_count(st)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIndexedToolCacheHit:
    async def test_exact_match_returns_cached(self, ask_scenario):
        resp, calls = await ask_scenario.turn(
            _ask("What are the state variables of the Vault contract?"),
        ).map_run(_response_and_calls)
        assert calls == 0
        assert "balances" in resp
        assert "Document-Ref:" in resp


class TestIndexedToolCacheMiss:
    async def test_novel_question_calls_answer_question(self, ask_scenario):
        resp, calls = await ask_scenario.turn(
            _ask("What ERC-20 tokens does the Vault interact with?"),
        ).map_run(_response_and_calls)
        assert calls == 1
        assert "FRESH ANSWER" in resp
        assert "Document-Ref:" in resp


class TestIndexedToolRecache:
    async def test_novel_then_reask_is_cached(self, ask_scenario):
        # First call: cache miss
        _, calls_after_first = await ask_scenario.turn(
            _ask("What ERC-20 tokens does the Vault interact with?"),
        ).map_run(_response_and_calls)
        assert calls_after_first == 1

        # Second call: same scenario (same index, answer was stored)
        resp, calls_after_second = await ask_scenario.turn(
            _ask("What ERC-20 tokens does the Vault interact with?"),
        ).map_run(_response_and_calls)
        # answer_question_calls is cumulative across both runs,
        # but should still be 1 — second call hit cache
        assert calls_after_second == 1
        assert "Document-Ref:" in resp
