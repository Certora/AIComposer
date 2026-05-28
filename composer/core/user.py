import os


def get_uid() -> str:
    uid = os.environ.get("AUTOPROVER_USER_ID")
    if not uid:
        uid = "_anonymous"
    return uid


def user_data_ns(uid: str | None = None) -> tuple[str, ...]:
    """Conventional namespace prefix for tenant-scoped data.

    Single source of truth: any per-user content (CVL research write
    layer, source-code-agent caches, etc.) is stored under
    ``("user_data", uid, …)``. Future refactors of the convention
    (per-org scope, per-engagement scope) only need to touch this
    function.
    """
    uid = get_uid() if not uid else uid
    return ("user_data", uid)
