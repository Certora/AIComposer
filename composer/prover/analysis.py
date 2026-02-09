from typing import Protocol

from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, BaseMessage, AnyMessage
from langchain_core.runnables import Runnable
from langchain_core.language_models.base import LanguageModelInput
from langgraph.config import get_stream_writer
from langgraph.config import get_store

from graphcore.graph import BoundLLM
from graphcore.utils import acached_invoke

from composer.core.state import AIComposerState
from composer.prover.ptypes import RuleResult
from composer.templates.loader import load_jinja_template
from composer.diagnostics.stream import ProgressUpdate

class AnalysisCB(Protocol):
    def get_cache(self) -> str | None:
        ...

    def put_cache(self, analysis: str):
        ...

    def on_analysis_start(self):
        pass

async def analyze_cex_raw(
        llm: Runnable[LanguageModelInput, BaseMessage],
        m: list[AnyMessage],
        rule: RuleResult,
        tool_call_id: str,
        cb: AnalysisCB | None = None
) -> str | None:
    if rule.status != "VIOLATED":
        return None
    
    if cb is not None and (res := cb.get_cache()) is not None:
        return res

    to_copy = m
    new_messages = to_copy.copy()

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
    if cb is not None:
        cb.on_analysis_start()
    
    res = await acached_invoke(llm, new_messages)
    if not isinstance(res, AIMessage):
        return None
    if cb is not None:
        cb.put_cache(res.text())
    return res.text()


async def analyze_cex(llm: BoundLLM, state: AIComposerState, rule: RuleResult, tool_call_id: str) -> str | None:
    if rule.status != "VIOLATED":
        return None
    
    writer = get_stream_writer()
    store = get_store()

    class CB():
        def put_cache(self, analysis: str):
            store.put(("cex", tool_call_id,), rule.path.pprint(), {"analysis": analysis})
        def get_cache(self) -> str | None:
            d = store.get(("cex", tool_call_id,), rule.path.pprint())
            if d is not None:
                return d.value["analysis"]
            return None
        def on_analysis_start(self):
            to_write: ProgressUpdate = {
                "type": "cex_analysis",
                "rule_name": rule.name
            }
            writer(to_write)

    cb = CB()
    return await analyze_cex_raw(
        llm, state["messages"], rule, tool_call_id, cb
    )
