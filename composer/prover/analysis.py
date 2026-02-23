from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, BaseMessage, AnyMessage
from langchain_core.runnables import Runnable
from langchain_core.language_models.base import LanguageModelInput

from graphcore.utils import acached_invoke

from composer.prover.ptypes import RuleResult
from composer.templates.loader import load_jinja_template


async def analyze_cex_raw(
        llm: Runnable[LanguageModelInput, BaseMessage],
        m: list[AnyMessage],
        rule: RuleResult,
        tool_call_id: str,
) -> str | None:
    if rule.status != "VIOLATED":
        return None

    new_messages = m.copy()

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
    return res.text()
