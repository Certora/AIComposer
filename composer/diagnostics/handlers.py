from typing import cast, TypedDict, TypeGuard, Optional

from composer.core.state import AIComposerState
from graphcore.graph import INITIAL_NODE, TOOL_RESULT_NODE, TOOLS_NODE
from composer.diagnostics.stream import AllUpdates, ProgressUpdate, AuditUpdate, UserUpdateTy, AuditUpdateTy
from composer.audit.db import AuditDB
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage

known_nodes = {INITIAL_NODE, TOOL_RESULT_NODE, TOOLS_NODE}

def normalize_content(s: str | list[str | dict]) -> list[dict]:
    l : list[str | dict]
    if isinstance(s, str):
        l = [s]
    else:
        l = s
    to_ret = []
    for r in l:
        if isinstance(r, str):
            to_ret.append({"type": "text", "text": r})
        else:
            to_ret.append(r)
    return to_ret

class CacheUsage(TypedDict):
    """
    Type for structured access to cache info
    """
    ephemeral_5m_input_tokens: int

class TokenUsage(TypedDict):
    """
    Type for structured access to token usage info
    """
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    cache_creation: CacheUsage

def summarize_update(
    state: dict
) -> None:
    for (node_name, v) in state.items():
        if node_name not in known_nodes:
            continue
        # this is actually a partial state, so we need to explicitly check membership
        state_update = cast(AIComposerState, v)
        printed = False

        def print_node() -> None:
            nonlocal printed
            if printed:
                return
            print(f"From node: {node_name}")
            printed = True

        if "messages" in state_update:
            print_node()
            for m in state_update["messages"]:
                match m:
                    case AIMessage():
                        buff = []
                        for c in normalize_content(m.content):
                            match c["type"]:
                                case "thinking":
                                    buff.append("Thinking...")
                                case "text":
                                    buff.append("Text: " + c["text"][:50] + "...")
                                case "tool_use":
                                    buff.append("Call tool: " + c["name"])
                                case _:
                                    buff.append("Unknown action: " + c["type"])
                        print("[AI turn]")
                        print("\n".join([f" > {t}" for t in buff]))
                        if isinstance(m.response_metadata, dict) and "usage" in m.response_metadata:
                            usage_data = cast(TokenUsage, m.response_metadata["usage"])
                            print(" -- Token stats:")
                            print(f" -> Cache read: {usage_data['cache_read_input_tokens']}")
                            print(f" -> Input: {usage_data['input_tokens']}")
                            print(f" -> Cache write: {usage_data['cache_creation']['ephemeral_5m_input_tokens']}")

                    case SystemMessage():
                        print("[System prompt]")
                    case HumanMessage():
                        print("[Initial prompt]")
                    case ToolMessage():
                        print("[Tool result]")
                    case _:
                        print(f"[Unhandled message {type(m)}]")
        if "vfs" in state_update:
            print_node()
            print("Put file(s):")
            for (k, _) in state_update["vfs"].items():
                print(f" > {k}")

# ++++++++++++++++++++++++
# Custom update handler
# +++++++++++++++++++++++++++

user_guard: set[UserUpdateTy] = {"cex_analysis", "prover_result", "prover_run"}

def is_user_update(x: AllUpdates) -> TypeGuard[ProgressUpdate]:
    return x["type"] in user_guard

def print_prover_updates(payload: ProgressUpdate) -> None:
    if payload["type"] == "cex_analysis":
        print(f"Analyzing CEX for rule {payload['rule_name']}")
    elif payload["type"] == "prover_result":
        print("Prover run complete, rule status:")
        print("\n".join([f" * {k}: {v}" for (k, v) in payload["status"].items()]))
    else:
        assert payload["type"] == "prover_run"
        print(f"Running prover with args: {' '.join(payload['args'])}")


audit_guard: set[AuditUpdateTy] = {"manual_search", "rule_result", "summarization"}

def is_audit_update(x: AllUpdates) -> TypeGuard[AuditUpdate]:
    return x["type"] in audit_guard

def handle_audit_update(db: AuditDB, upd: AuditUpdate, thread_id: str) -> None:
    match upd["type"]:
        case "manual_search":
            db.add_manual_result(
                thread_id=thread_id,
                tool_id=upd["tool_id"],
                ref=upd["ref"]
            )
        case "rule_result":
            db.add_rule_result(
                thread_id=thread_id,
                analysis=upd["analysis"],
                result=upd["status"],
                rule_name=upd["rule"],
                tool_id=upd["tool_id"]
            )
        case "summarization":
            db.register_summary(
                thread_id=thread_id,
                checkpoint_id=upd["checkpoint_id"],
                summary=upd["summary"]
            )
        

def handle_custom_update(p: AllUpdates, thread_id: str, audit_db: Optional[AuditDB]) -> None:
    if is_user_update(p):
        print_prover_updates(p)
    elif is_audit_update(p) and audit_db is not None:
        handle_audit_update(audit_db, p, thread_id=thread_id)
