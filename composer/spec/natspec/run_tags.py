"""Tag schema for natspec pipeline runs.

Records the inputs that drive the natspec cache-key layout into the
``RunMeta.tags`` slot at run start so downstream tooling
(``cache-natspec``) can rehydrate the namespaces from a run id alone.

This is the *canonical* shape — both the writer (``tui-natspec``) and
the reader (``cache-natspec``) reference this model; round-trip via
``model_dump()`` into ``RunMeta.tags`` and ``model_validate(tags)`` back.
"""

from pydantic import BaseModel


class NatspecRunTags(BaseModel):
    root_thread_id: str
    doc_digest: str
    cache_namespace: str | None
    memory_namespace: str | None
    from_source: bool
    interactive: bool
