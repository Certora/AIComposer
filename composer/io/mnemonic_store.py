import json
import logging
import uuid

from langgraph.store.base import BaseStore, PutOp
from langgraph.store.postgres import AsyncPostgresStore

from composer.ui.mnemonic import mnemonic_id


_logger = logging.getLogger("composer.mnemonic")


def _mnem_ns(base_ns: tuple[str, ...]) -> tuple[str, ...]:
    return base_ns + ("mnem",)

def _thread_ns(base_ns: tuple[str, ...]) -> tuple[str, ...]:
    return base_ns + ("thread",)


def _ns_text(ns: tuple[str, ...]) -> str:
    # Mirrors langgraph.store.postgres.base._namespace_to_text. Inlined
    # to avoid importing a private symbol; the joining rule is stable.
    return ".".join(ns)


# INSERT ... ON CONFLICT DO NOTHING is the Postgres atomic put-if-absent
# primitive. ``rowcount == 1`` iff this caller won the race for (prefix, key);
# losers retry with a different candidate. Schema matches the langgraph
# store migrations (prefix, key, value, created_at, updated_at, ttl_minutes).
_CLAIM_SQL = """
INSERT INTO store (prefix, key, value, created_at, updated_at, ttl_minutes)
VALUES (%s, %s, %s::jsonb, NOW(), NOW(), NULL)
ON CONFLICT (prefix, key) DO NOTHING
"""


async def _assign_pgsql_mnemonic(
    tid: str,
    store: AsyncPostgresStore,
    mnemonic_ns: tuple[str, ...]
) -> str:
    """CAS-based mnemonic assignment for AsyncPostgresStore.

    Each iteration mints a candidate id and tries to claim the
    (mnem-ns, id) slot via ``INSERT ... ON CONFLICT DO NOTHING``. Only
    one concurrent caller wins a given id; losers retry. After nine
    vanity-id attempts, fall back to appending 12 hex chars from a
    UUID, which is collision-resistant by birthday bound.

    Goes through ``store._cursor()`` to share the same pool / lock
    plumbing as the rest of the store's writes. ``_cursor`` is a
    single-underscore-private of langgraph; the shape has been stable
    across minor versions and we control our pin.
    """
    ms = _mnem_ns(mnemonic_ns)
    ts = _thread_ns(mnemonic_ns)

    # Fast path: an earlier call already assigned a mnemonic for this tid.
    existing = await store.aget(ts, tid)
    if existing is not None:
        return existing.value["mnem"]

    ms_prefix = _ns_text(ms)
    payload = json.dumps({"tid": tid})

    for i in range(10):
        cand = mnemonic_id()
        if i == 9:
            cand = cand + "-" + uuid.uuid4().hex[:12]

        async with store._cursor() as cur:
            await cur.execute(_CLAIM_SQL, (ms_prefix, cand, payload))
            claimed = cur.rowcount == 1

        if not claimed:
            continue

        # Won the race on the ms-side. Two concurrent callers for the same
        # ``tid`` could each have claimed a distinct mnem; on the ts-side
        # last-write-wins and the other mnem is left orphaned. Tolerable.
        await store.aput(ts, tid, {"mnem": cand})
        _logger.info(f"assigned mnemonic {cand} to thread {tid}")
        return cand

    assert False, "Somehow, implausibly, failed to allocate a fresh UID"

async def assign_mnemonic(
    tid: str,
    store: BaseStore,
    mnemonic_ns: tuple[str, ...]
) -> str:
    if isinstance(store, AsyncPostgresStore):
        return await _assign_pgsql_mnemonic(tid, store, mnemonic_ns)

    ms = _mnem_ns(mnemonic_ns)
    ts = _thread_ns(mnemonic_ns)

    # yes, this is racy, no I don't care
    mnem = await store.aget(ts, tid)
    if mnem is not None:
        return mnem.value["mnem"]
    
    for i in range(0, 10):
        id = mnemonic_id()
        # we give up!
        if i == 9:
            id = id + "-" + uuid.uuid4().hex[:12]
        taken = await store.aget(ms, id)
        if taken:
            continue

        await store.abatch([
            PutOp(
                namespace=ms,
                key=id,
                value={"tid": tid}
            ),
            PutOp(
                namespace=ts,
                key=tid,
                value={"mnem": id}
            )
        ])
        _logger.info(f"assigned mnemonic {id} to thread {tid}")
        return id

    assert False, "Somehow, implausibly, failed to allocate a fresh UID"