
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langgraph.config import get_stream_writer
from langgraph.config import get_store

from graphcore.graph import BoundLLM
from graphcore.utils import acached_invoke

from composer.core.state import AIComposerState
from composer.prover.ptypes import RuleResult
from composer.templates.loader import load_jinja_template
from composer.diagnostics.stream import ProgressUpdate

async def analyze_cex(llm: BoundLLM, state: AIComposerState, rule: RuleResult, tool_call_id: str) -> str | None:
    if rule.status != "VIOLATED":
        return None
    to_copy = state["messages"]
    new_messages = to_copy.copy()
    writer = get_stream_writer()
    store = get_store()
    d = store.get(("cex", tool_call_id,), rule.name)
    if d is not None:
        return d.value["analysis"]

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
    res = await acached_invoke(llm, new_messages)
    if not isinstance(res, AIMessage):
        return None
    store.put(("cex", tool_call_id,), rule.name, {"analysis": res.text()})
    return res.text()
