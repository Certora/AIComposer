from verisafe.core.state import CryptoStateGen
from graphcore.graph import BoundLLM
from verisafe.prover.ptypes import RuleResult
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from graphcore.utils import cached_invoke
from verisafe.templates.loader import load_jinja_template
from langgraph.config import get_stream_writer
from verisafe.diagnostics.stream import ProgressUpdate

def analyze_cex(llm: BoundLLM, state: CryptoStateGen, rule: RuleResult, tool_call_id: str) -> str | None:
    if rule.status != "VIOLATED":
        return None
    to_copy = state["messages"]
    new_messages = to_copy.copy()
    writer = get_stream_writer()
    to_write: ProgressUpdate = {
        "type": "cex_analysis",
        "rule_name": rule.name
    }
    writer(to_write)
    new_messages.append(
        ToolMessage(
            tool_call_id=tool_call_id,
            content=f"""
The Certora Prover found a violation for the rule {rule.name}, with the following counter example:
{rule.cex_dump}
"""
        )
    )
    new_messages.append(
        HumanMessage(
            content=load_jinja_template("cex_instructions.j2", rule_name=rule.name)
        )
    )
    res = cached_invoke(llm, new_messages)
    if not isinstance(res, AIMessage):
        return None
    content = res.content
    content_list = content
    if not isinstance(content_list, list):
        content_list = [content_list]
    for i in range(len(content_list) - 1, -1, -1):
        m = content_list[i]
        if isinstance(m, str):
            return m
        if m.get("type", None) != "text":
            continue
        return m.get("text", None)
    return None
