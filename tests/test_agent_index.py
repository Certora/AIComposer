"""Unit tests for AgentIndex layered read/write semantics.

Uses langgraph's InMemoryStore — no testcontainers / Postgres required.
Vector embedding goes through the MockSentenceTransformer in conftest:
registered Q/A pairs land at deterministic random unit vectors;
unknown text falls back to hash-based vectors. That's enough to drive
all the layer-targeting and isolation behaviors without relying on real
semantic similarity.
"""

from __future__ import annotations

from typing import Callable

import pytest
import pytest_asyncio

from langgraph.store.memory import InMemoryStore

from composer.kb.knowledge_base import DefaultEmbedder
from composer.spec.agent_index import (
    AgentIndex,
    AgentIndexConfig,
    agent_index_config_from_env,
    user_data_ns,
)

from .conftest import QnATransformer, EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_NS: tuple[str, ...] = ("cvl_research", "cached")
ALICE = "alice"
BOB = "bob"


@pytest_asyncio.fixture
async def store(qna_factory: Callable[[], QnATransformer]) -> InMemoryStore:
    qna = qna_factory()
    # Register a few texts so their vectors are stable across calls.
    # Unregistered texts get hash-deterministic random vectors.
    for q in ("Q_shared", "Q_alice", "Q_bob", "What color is the sky?"):
        qna.register(q, [q])
    return InMemoryStore(
        index={
            "embed": DefaultEmbedder(model=qna.as_transformer),
            "dims": EMBEDDING_DIM,
            "fields": None,
        }
    )


def trusted_config() -> AgentIndexConfig:
    return AgentIndexConfig(base_layer=DATA_NS)


def tiered_config(uid: str) -> AgentIndexConfig:
    return AgentIndexConfig(
        base_layer=DATA_NS,
        write_layer=user_data_ns(uid) + DATA_NS,
    )


def readonly_config() -> AgentIndexConfig:
    return AgentIndexConfig(base_layer=DATA_NS, read_only=True)


# ---------------------------------------------------------------------------
# user_data_ns convention
# ---------------------------------------------------------------------------


def test_user_data_ns_shape():
    assert user_data_ns("alice") == ("user_data", "alice")


# ---------------------------------------------------------------------------
# Env → config
# ---------------------------------------------------------------------------


class TestConfigFromEnv:
    """The env helper's mode dispatch is the only place the "user_data"
    convention is encoded for the CVL-research path. Verify each branch
    yields the expected fully-specified AgentIndexConfig."""

    def test_default_is_trusted(self, monkeypatch):
        monkeypatch.delenv("AUTOPROVER_AGENT_INDEX_MODE", raising=False)
        monkeypatch.delenv("AUTOPROVER_USER_ID", raising=False)
        cfg = agent_index_config_from_env(DATA_NS)
        assert cfg == AgentIndexConfig(base_layer=DATA_NS)

    def test_trusted_explicit(self, monkeypatch):
        monkeypatch.setenv("AUTOPROVER_AGENT_INDEX_MODE", "trusted")
        cfg = agent_index_config_from_env(DATA_NS)
        assert cfg.write_layer is None
        assert cfg.read_only is False
        assert cfg.base_layer == DATA_NS

    def test_readonly(self, monkeypatch):
        monkeypatch.setenv("AUTOPROVER_AGENT_INDEX_MODE", "readonly")
        cfg = agent_index_config_from_env(DATA_NS)
        assert cfg.read_only is True
        assert cfg.write_layer is None
        assert cfg.base_layer == DATA_NS

    def test_tiered_with_uid(self, monkeypatch):
        monkeypatch.setenv("AUTOPROVER_AGENT_INDEX_MODE", "tiered")
        monkeypatch.setenv("AUTOPROVER_USER_ID", ALICE)
        cfg = agent_index_config_from_env(DATA_NS)
        assert cfg.base_layer == DATA_NS
        assert cfg.write_layer == ("user_data", ALICE) + DATA_NS
        assert cfg.read_only is False

    def test_tiered_without_uid_raises(self, monkeypatch):
        monkeypatch.setenv("AUTOPROVER_AGENT_INDEX_MODE", "tiered")
        monkeypatch.delenv("AUTOPROVER_USER_ID", raising=False)
        with pytest.raises(ValueError, match="AUTOPROVER_USER_ID"):
            agent_index_config_from_env(DATA_NS)

    def test_unknown_mode_raises(self, monkeypatch):
        monkeypatch.setenv("AUTOPROVER_AGENT_INDEX_MODE", "garbage")
        with pytest.raises(ValueError, match="Unknown"):
            agent_index_config_from_env(DATA_NS)


# ---------------------------------------------------------------------------
# Write targeting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWriteTargeting:
    """``aput`` lands in the right namespace per mode."""

    async def test_trusted_writes_to_base(self, store):
        idx = AgentIndex(store, trusted_config())
        key = await idx.aput(question="Q1", answer="A1")
        assert key is not None
        item = await store.aget(DATA_NS, key)
        assert item is not None
        assert item.value["answer"] == "A1"

    async def test_tiered_writes_to_overlay_only(self, store):
        idx = AgentIndex(store, tiered_config(ALICE))
        key = await idx.aput(question="Q1", answer="A1")
        assert key is not None
        overlay = user_data_ns(ALICE) + DATA_NS
        item = await store.aget(overlay, key)
        assert item is not None
        assert item.value["answer"] == "A1"
        # Base does NOT get the write.
        assert await store.aget(DATA_NS, key) is None

    async def test_readonly_drops_writes(self, store):
        idx = AgentIndex(store, readonly_config())
        ref = await idx.aput(question="Q1", answer="A1")
        assert ref is None
        # Nothing landed anywhere reachable.
        key = idx._question_key("Q1")
        assert await store.aget(DATA_NS, key) is None
        assert await store.aget(user_data_ns(ALICE) + DATA_NS, key) is None

    async def test_first_write_wins_per_layer(self, store):
        idx = AgentIndex(store, tiered_config(ALICE))
        k1 = await idx.aput(question="Q1", answer="A1")
        k2 = await idx.aput(question="Q1", answer="A2 (newer, ignored)")
        assert k1 == k2
        item = await store.aget(user_data_ns(ALICE) + DATA_NS, k1)
        assert item.value["answer"] == "A1"

    async def test_overlay_and_base_can_hold_same_key(self, store):
        # The first-write-wins rule applies *within* a layer, not across.
        # A trusted write into base and a tiered write into overlay both
        # land cleanly even when they share a normalized-question key.
        trusted_idx = AgentIndex(store, trusted_config())
        alice_idx = AgentIndex(store, tiered_config(ALICE))
        k_base = await trusted_idx.aput(question="Q1", answer="A_base")
        k_overlay = await alice_idx.aput(question="Q1", answer="A_overlay")
        assert k_base == k_overlay
        assert (await store.aget(DATA_NS, k_base)).value["answer"] == "A_base"
        assert (
            await store.aget(user_data_ns(ALICE) + DATA_NS, k_overlay)
        ).value["answer"] == "A_overlay"


# ---------------------------------------------------------------------------
# Read order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReadOrder:
    async def test_overlay_shadows_base(self, store):
        trusted_idx = AgentIndex(store, trusted_config())
        gkey = await trusted_idx.aput(question="Q1", answer="A_base")

        alice_idx = AgentIndex(store, tiered_config(ALICE))
        akey = await alice_idx.aput(question="Q1", answer="A_overlay")
        assert gkey == akey

        # Alice sees the overlay.
        result = await alice_idx.aget(gkey)
        assert result is not None
        assert result["answer"] == "A_overlay"

        # Trusted (base-only) sees the base.
        result = await trusted_idx.aget(gkey)
        assert result is not None
        assert result["answer"] == "A_base"

    async def test_overlay_miss_falls_through_to_base(self, store):
        # Seed base only.
        trusted_idx = AgentIndex(store, trusted_config())
        gkey = await trusted_idx.aput(question="Q1", answer="A_base")

        # Alice's overlay is empty for Q1; aget falls through.
        alice_idx = AgentIndex(store, tiered_config(ALICE))
        result = await alice_idx.aget(gkey)
        assert result is not None
        assert result["answer"] == "A_base"

    async def test_aget_misses_when_neither_layer_has_it(self, store):
        idx = AgentIndex(store, tiered_config(ALICE))
        fake_key = idx._question_key("Never written")
        assert await idx.aget(fake_key) is None


# ---------------------------------------------------------------------------
# Search semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSearch:
    async def test_exact_key_returns_keyed_result(self, store):
        idx = AgentIndex(store, trusted_config())
        await idx.aput(question="What color is the sky?", answer="Blue.")
        res = await idx.asearch("What color is the sky?")
        # KeyedAgentResult is a dict, the vector-miss path returns a list.
        assert isinstance(res, dict)
        assert res["answer"] == "Blue."
        assert "ref_string" in res

    async def test_vector_miss_merges_both_layers(self, store):
        # Different normalized-question keys → exact-key path misses
        # for the search query, vector path runs across both pools.
        trusted_idx = AgentIndex(store, trusted_config())
        await trusted_idx.aput(question="Q_shared", answer="A_base")
        alice_idx = AgentIndex(store, tiered_config(ALICE))
        await alice_idx.aput(question="Q_alice", answer="A_alice")

        results = await alice_idx.asearch("Q_unknown_query_for_vector_path")
        assert isinstance(results, list)
        answers = {r["answer"] for r in results}
        # Both pools' contents are reachable via the merge.
        assert "A_base" in answers
        assert "A_alice" in answers


# ---------------------------------------------------------------------------
# Cross-tenant isolation (the bug the layout fix addresses)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCrossTenantIsolation:
    """The sibling-namespace layout (``user_data_ns(uid) + base``,
    placed at the *outermost* level, not as a descendant of the base)
    is the reason cross-tenant search leakage is impossible. Verify it
    holds for the langgraph backend's prefix-matching ``asearch``."""

    async def test_tenant_overlay_invisible_to_other_tenant(self, store):
        alice_idx = AgentIndex(store, tiered_config(ALICE))
        await alice_idx.aput(question="Q_alice", answer="A_alice_secret")

        bob_idx = AgentIndex(store, tiered_config(BOB))
        # Exact-question search → exact-key path probes Bob's overlay
        # and the base. Neither has the entry. Vector path then searches
        # both Bob's overlay and the base — Alice's overlay is *not* in
        # Bob's read pools, so her writes don't surface.
        res = await bob_idx.asearch("Q_alice")
        if isinstance(res, dict):
            assert res["answer"] != "A_alice_secret"
        else:
            answers = {r["answer"] for r in res}
            assert "A_alice_secret" not in answers

    async def test_tenant_overlay_invisible_to_trusted_reader(self, store):
        alice_idx = AgentIndex(store, tiered_config(ALICE))
        await alice_idx.aput(question="Q_alice", answer="A_alice_secret")

        # A trusted-mode index reads only the bare base_layer. The
        # backend's prefix-LIKE / tuple-prefix search must NOT reach
        # into Alice's overlay (which lives at user_data/alice/...,
        # not at base_layer/...).
        trusted_idx = AgentIndex(store, trusted_config())
        res = await trusted_idx.asearch("Q_alice")
        if isinstance(res, dict):
            assert res["answer"] != "A_alice_secret"
        else:
            answers = {r["answer"] for r in res}
            assert "A_alice_secret" not in answers
