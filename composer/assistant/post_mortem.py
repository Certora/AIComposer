"""Post-mortem analysis tool for extracting decisions from failed/completed workflows."""

import asyncio
from dataclasses import dataclass

from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models import BaseChatModel

from graphcore.tools.memory import PostgresMemoryBackend
from graphcore.tools.schemas import WithAsyncImplementation
from langgraph.runtime import get_runtime

from composer.assistant.types import OrchestratorContext
from composer.templates.loader import load_jinja_template
from composer.workflow.factories import get_memory_ns
from composer.workflow.services import get_checkpointer, get_memory


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

class PostMortemArgs(BaseModel):
    codegen_thread_id: str = Field(description="Thread ID of the main codegen workflow to analyze")
    natreq_thread_id: str | None = Field(description="Thread ID of the requirements extraction workflow (omit if requirements were cached)", default=None)
    memory_namespace: str = Field(description="Root memory namespace to write extracted decisions to")


# ---------------------------------------------------------------------------
# HITL interaction extraction
# ---------------------------------------------------------------------------

_HITL_TOOLS = frozenset({
    "human_in_the_loop",
    "propose_spec_change",
    "requirement_relaxation_request",
})


@dataclass
class _HITLInteraction:
    """A single human-in-the-loop interaction: question + response."""
    tool_name: str
    question: dict
    response: str


@dataclass
class _ConversationData:
    """Extracted conversation data: HITL interactions + summarization context."""
    interactions: list[_HITLInteraction]
    summaries: list[str]


def _extract_conversation_data(messages: list) -> _ConversationData:
    """Extract HITL interaction pairs and summarization context from messages."""
    # Index ToolMessages by tool_call_id for fast lookup
    tool_responses: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_responses[msg.tool_call_id] = msg.text

    interactions: list[_HITLInteraction] = []
    summaries: list[str] = []

    for msg in messages:
        # Capture summarization resume messages (contain prior decision summaries)
        if isinstance(msg, HumanMessage):
            tag = getattr(msg, "display_tag", None)
            if tag == "resume":
                text = msg.text
                if text.strip():
                    summaries.append(text)
            continue

        if not isinstance(msg, AIMessage):
            continue

        for tc in msg.tool_calls:
            name = tc.get("name", "")
            if name not in _HITL_TOOLS:
                continue
            call_id = tc.get("id", "")
            if call_id is None:
                continue
            response = tool_responses.get(call_id)
            if response is None:
                continue
            interactions.append(_HITLInteraction(
                tool_name=name,
                question=tc.get("args", {}),
                response=response,
            ))

    return _ConversationData(interactions=interactions, summaries=summaries)


def _format_conversation_data(data: _ConversationData) -> str | None:
    """Format extracted data into a readable transcript. Returns None if empty."""
    parts: list[str] = []

    if data.summaries:
        parts.append("## Prior Conversation Summaries\n"
                      "These summaries were generated when the conversation was "
                      "compacted. They may contain decisions from earlier interactions.")
        for i, summary in enumerate(data.summaries, 1):
            parts.append(f"### Summary {i}\n{summary}")

    if data.interactions:
        parts.append("## Human Interactions")
        for i, interaction in enumerate(data.interactions, 1):
            match interaction.tool_name:
                case "human_in_the_loop":
                    question = interaction.question.get("question", "")
                    context = interaction.question.get("context", "")
                    parts.append(
                        f"### Interaction {i}: Question\n"
                        f"**Context:** {context}\n"
                        f"**Question:** {question}\n"
                        f"**Human response:** {interaction.response}"
                    )
                case "propose_spec_change":
                    explanation = interaction.question.get("explanation", "")
                    parts.append(
                        f"### Interaction {i}: Spec Change Proposal\n"
                        f"**Explanation:** {explanation}\n"
                        f"**Human response:** {interaction.response}"
                    )
                case "requirement_relaxation_request":
                    context = interaction.question.get("context", "")
                    req_text = interaction.question.get("req_text", "")
                    parts.append(
                        f"### Interaction {i}: Requirement Relaxation\n"
                        f"**Requirement:** {req_text}\n"
                        f"**Context:** {context}\n"
                        f"**Human response:** {interaction.response}"
                    )
                case _:
                    parts.append(
                        f"### Interaction {i}: {interaction.tool_name}\n"
                        f"**Args:** {interaction.question}\n"
                        f"**Human response:** {interaction.response}"
                    )

    if not parts:
        return None
    return "\n\n".join(parts)


def _dump_memory(backend: PostgresMemoryBackend) -> str | None:
    """Read all files from a memory backend and concatenate them."""
    parts: list[str] = []

    def _walk(path: str) -> None:
        for name, is_dir in backend.list_dir(path):
            child = f"{path}/{name}" if path != "/memories" else f"/memories/{name}"
            if is_dir:
                _walk(child)
            else:
                content = backend.view(child, None)
                parts.append(f"### {child}\n{content}")

    try:
        _walk("/memories")
    except Exception:
        return None

    return "\n\n".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Single-thread extraction
# ---------------------------------------------------------------------------

async def _extract_one(
    thread_id: str,
    memory_ns: str,
    system_template: str,
    user_template: str,
    llm: BaseChatModel,
) -> str:
    """Run extraction for a single thread and write results to memory."""
    from langchain_core.messages import SystemMessage

    checkpointer = get_checkpointer()
    checkpoint_tuple = checkpointer.get_tuple({"configurable": {"thread_id": thread_id}})
    if checkpoint_tuple is None:
        return f"No checkpoint found for thread {thread_id}."

    messages = checkpoint_tuple.checkpoint["channel_values"].get("messages", [])
    if not messages:
        return f"No messages found in thread {thread_id}."

    data = _extract_conversation_data(messages)
    conversation = _format_conversation_data(data)
    if conversation is None:
        return f"No human interactions or summaries found in thread {thread_id}."

    backend = get_memory(memory_ns)
    existing_memories = _dump_memory(backend)

    system_text = load_jinja_template(system_template)
    user_text = load_jinja_template(
        user_template,
        existing_memories=existing_memories,
        conversation=conversation,
    )

    result = await llm.ainvoke([  # type: ignore[union-attr]
        SystemMessage(content=system_text),
        HumanMessage(content=user_text),
    ])
    extracted = result.text

    if extracted.strip():
        backend.create("/memories/decisions.md", extracted)

    n_interactions = len(data.interactions)
    n_summaries = len(data.summaries)
    return (
        f"Extracted decisions from thread {thread_id}: "
        f"{n_interactions} interaction(s), {n_summaries} summary context(s)."
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class PostMortemTool(PostMortemArgs, WithAsyncImplementation[str]):
    """Analyze completed/failed workflows to extract key decisions into persistent memory.

    Runs post-mortem extraction on the codegen and (optionally) natreq
    conversations, writing results to the appropriate memory sub-namespaces.
    """

    async def run(self) -> str:
        ctx = get_runtime(OrchestratorContext).context

        codegen_ns = get_memory_ns(self.memory_namespace, "composer")
        tasks = [
            _extract_one(
                self.codegen_thread_id,
                codegen_ns,
                "post_mortem_codegen_system.j2",
                "post_mortem_codegen.j2",
                ctx.llm,
            )
        ]

        if self.natreq_thread_id is not None:
            natreq_ns = get_memory_ns(self.memory_namespace, "natreq")
            tasks.append(
                _extract_one(
                    self.natreq_thread_id,
                    natreq_ns,
                    "post_mortem_natreq_system.j2",
                    "post_mortem_natreq.j2",
                    ctx.llm,
                )
            )

        results = await asyncio.gather(*tasks)
        return "\n".join(results)
