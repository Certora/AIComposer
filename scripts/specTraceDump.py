"""
Trace dump utility for the natspec (auto_spec.py) workflow.
Extracts execution traces from the checkpointer and renders them as interactive HTML.

Usage: python scripts/specTraceDump.py <thread_id> <output_file>
"""
import sys

if __name__ != "__main__":
    raise RuntimeError("This is a script only module")

import bind as _

import json
from dataclasses import dataclass
from typing import List, cast, TypedDict, Literal, Annotated, Union
from langgraph.checkpoint.base import CheckpointTuple
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage, SystemMessage
from pydantic import Discriminator

from composer.workflow.factories import get_checkpointer
from composer.templates.loader import load_jinja_template

# Import common utilities and base types
from traceCommon import (
    AbstractStep, AIStepMessage, AIStep, QuestionStep, FreshFile, Diff, FileUpdate,
    PutFileStep, SummarizationStep, Command, VFSInteraction, VFSStep,
    VFSManager, compute_diff
)


# ========================================
# Natspec-Specific Step Type Definitions
# ========================================

class InitialStep(AbstractStep[Literal["initial"]]):
    """Initial step showing the design document"""
    document: str


class FeedbackItem(TypedDict):
    """Single feedback item from guidelines judge"""
    severity: Literal["Critical", "Advice", "Informational"]
    description: str
    code_reference: str
    guideline_ref: str


class GuidelinesStep(AbstractStep[Literal["guidelines"]]):
    """Feedback from guidelines_judge tool"""
    feedback_items: List[FeedbackItem]


class SuggestionStep(AbstractStep[Literal["suggestion"]]):
    """Suggestions from suggestion_oracle tool"""
    suggestions: List[str]


class TypecheckStep(AbstractStep[Literal["typecheck"]]):
    """Results from typecheck_spec tool"""
    success: bool
    output: str


class ResultStep(AbstractStep[Literal["result"]]):
    """Final generation complete"""
    expected_solc: str
    expected_contract_name: str
    implementation_notes: str


class SearchResult(TypedDict):
    """Single search result from cvl_manual_search"""
    header: str
    content: str
    similarity: float


class SearchStep(AbstractStep[Literal["search"]]):
    """CVL manual search results"""
    query: str
    results: List[SearchResult]


Steps = Annotated[Union[
    AIStep,
    InitialStep,
    GuidelinesStep,
    SuggestionStep,
    TypecheckStep,
    PutFileStep,
    QuestionStep,
    ResultStep,
    SummarizationStep,
    VFSStep,
    SearchStep
], Discriminator("type")]


# ========================================
# View Event Types
# ========================================

@dataclass
class ViewEvent:
    """Base class for events in the unified message view"""
    pass


@dataclass
class MessageEvent(ViewEvent):
    """A new message appeared in the checkpoint"""
    message: BaseMessage


@dataclass
class SpecUpdatedEvent(ViewEvent):
    """curr_spec channel value changed"""
    new_spec: str


@dataclass
class IntfUpdatedEvent(ViewEvent):
    """curr_intf channel value changed"""
    new_intf: str


@dataclass
class SummarizationEvent(ViewEvent):
    """Summarization occurred (message count decreased)"""
    summary_text: str


# ========================================
# Event Queue
# ========================================

class EventQueue:
    """Queue for processing view events sequentially"""

    def __init__(self, events: List[ViewEvent]):
        self.events = events
        self.i = 0

    def peek(self) -> ViewEvent | None:
        """Look at next event without consuming it"""
        if self.i >= len(self.events):
            return None
        return self.events[self.i]

    def take(self) -> ViewEvent:
        """Consume and return next event"""
        event = self.events[self.i]
        self.i += 1
        return event

    def has_next(self) -> bool:
        """Check if more events available"""
        return self.i < len(self.events)

    def skip_until_message(self) -> None:
        """Skip non-message events (state updates consumed elsewhere)"""
        while self.has_next():
            if isinstance(self.peek(), MessageEvent):
                break
            self.i += 1


# ========================================
# Global State
# ========================================

thread_id = sys.argv[1]
output_file = sys.argv[2]

checkpointer = get_checkpointer()


# ========================================
# Checkpoint Processing
# ========================================

def get_checkpoint_chain(start: CheckpointTuple) -> List[CheckpointTuple]:
    """
    Build list of checkpoints from start back to root.
    Returns in chronological order (oldest first).
    """
    chain: List[CheckpointTuple] = []
    current: CheckpointTuple | None = start

    while current is not None:
        chain.append(current)
        if current.parent_config is not None:
            current = checkpointer.get_tuple(current.parent_config)
        else:
            current = None

    chain.reverse()  # Oldest first
    return chain


def get_checkpoint_state(checkpoint: CheckpointTuple) -> tuple[List[BaseMessage], str | None, str | None]:
    """Extract messages, curr_spec, curr_intf from checkpoint"""
    channel_values = checkpoint.checkpoint.get("channel_values", {})
    messages = cast(List[BaseMessage], channel_values.get("messages", []))
    curr_spec = channel_values.get("curr_spec")
    curr_intf = channel_values.get("curr_intf")
    return (messages, curr_spec, curr_intf)


def extract_summary_text(messages: List[BaseMessage]) -> str:
    """Extract summary text from messages after summarization"""
    # Summary appears in messages[2] (HumanMessage) after summarization
    if len(messages) >= 3:
        summary_msg = messages[2]
        if isinstance(summary_msg, HumanMessage):
            content = summary_msg.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list) and len(content) > 0:
                first = content[0]
                if isinstance(first, str):
                    return first
                elif isinstance(first, dict) and "text" in first:
                    return cast(str, first["text"])
    return "(Summary text not found)"


def build_message_view(checkpoints: List[CheckpointTuple]) -> List[ViewEvent]:
    """
    Build a unified event stream from checkpoints.

    Uses message IDs for deduplication and inserts explicit markers
    for state changes and summarization events.
    """
    seen_message_ids: set[str] = set()
    prev_spec: str | None = None
    prev_intf: str | None = None
    prev_message_count: int = 0
    events: List[ViewEvent] = []

    for checkpoint in checkpoints:
        messages, curr_spec, curr_intf = get_checkpoint_state(checkpoint)

        # Detect summarization (message count decreased)
        if len(messages) < prev_message_count:
            summary_text = extract_summary_text(messages)
            events.append(SummarizationEvent(summary_text=summary_text))
            # Reset seen IDs since history was truncated
            seen_message_ids = {m.id for m in messages if m.id is not None}


        # Add new messages (deduplicated by ID)
        for msg in messages:
            msg_id = msg.id
            if msg_id is not None and msg_id not in seen_message_ids:
                seen_message_ids.add(msg_id)
                events.append(MessageEvent(message=msg))

        # Detect curr_spec change
        if curr_spec != prev_spec and curr_spec is not None:
            events.append(SpecUpdatedEvent(new_spec=curr_spec))
            prev_spec = curr_spec

        # Detect curr_intf change
        if curr_intf != prev_intf and curr_intf is not None:
            events.append(IntfUpdatedEvent(new_intf=curr_intf))
            prev_intf = curr_intf

        prev_message_count = len(messages)

    return events


# ========================================
# XML Parsing for Guidelines Judge
# ========================================

def parse_guidelines_feedback(text: str) -> List[FeedbackItem]:
    """
    Parse XML feedback from guidelines_judge tool.
    Handles unescaped XML content with fallback to manual parsing.
    """
    import xml.etree.ElementTree as ET

    if not text.strip():
        return []

    items: List[FeedbackItem] = []
    parts = text.split('</feedback>')

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if not part.endswith('</feedback>'):
            part += '</feedback>'

        start_idx = part.find('<feedback>')
        if start_idx == -1:
            continue
        part = part[start_idx:]

        severity: str = "Informational"
        description: str = ""
        code_reference: str = ""
        guideline_ref: str = ""

        try:
            root = ET.fromstring(part)

            sev_el = root.find('severity')
            desc_el = root.find('description')
            code_el = root.find('code_ref')
            guide_el = root.find('guideline')

            if sev_el is not None and sev_el.text:
                severity = sev_el.text
            if desc_el is not None and desc_el.text:
                description = desc_el.text
            if code_el is not None and code_el.text:
                code_reference = code_el.text
            if guide_el is not None and guide_el.text:
                guideline_ref = guide_el.text

        except ET.ParseError:
            def extract_tag(s: str, tag: str) -> str:
                start = s.find(f'<{tag}>') + len(f'<{tag}>')
                end = s.find(f'</{tag}>')
                if start > len(f'<{tag}>') - 1 and end != -1:
                    return s[start:end]
                return ""

            severity = extract_tag(part, 'severity') or "Informational"
            description = extract_tag(part, 'description')
            code_reference = extract_tag(part, 'code_ref')
            guideline_ref = extract_tag(part, 'guideline')

        if severity not in ["Critical", "Advice", "Informational"]:
            severity = "Informational"

        items.append(FeedbackItem(
            severity=cast(Literal["Critical", "Advice", "Informational"], severity),
            description=description,
            code_reference=code_reference,
            guideline_ref=guideline_ref
        ))

    return items


# ========================================
# Tool Handlers (Event-Based)
# ========================================

def consume_tool_message(queue: EventQueue, tool_call_id: str) -> ToolMessage:
    """Consume and return the ToolMessage for a given tool call"""
    event = queue.take()
    assert isinstance(event, MessageEvent), f"Expected MessageEvent, got {type(event)}"
    msg = event.message
    assert isinstance(msg, ToolMessage), f"Expected ToolMessage, got {type(msg)}"
    assert msg.tool_call_id == tool_call_id, f"Tool call ID mismatch: {msg.tool_call_id} != {tool_call_id}"
    return msg


def handle_guidelines_judge(queue: EventQueue, tool_call_id: str, vfs_version: int) -> GuidelinesStep:
    """Handle guidelines_judge tool"""
    tool_msg = consume_tool_message(queue, tool_call_id)
    feedback_items = parse_guidelines_feedback(tool_msg.text())

    return GuidelinesStep(
        type="guidelines",
        vfs_snapshot=vfs_version,
        feedback_items=feedback_items
    )


def handle_suggestion_oracle(queue: EventQueue, tool_call_id: str, vfs_version: int) -> SuggestionStep:
    """Handle suggestion_oracle tool"""
    tool_msg = consume_tool_message(queue, tool_call_id)
    text = tool_msg.text().strip()

    suggestions = [s.strip() for s in text.split('\n') if s.strip()] if text else []

    return SuggestionStep(
        type="suggestion",
        vfs_snapshot=vfs_version,
        suggestions=suggestions
    )


def handle_typecheck_spec(queue: EventQueue, tool_call_id: str, vfs_version: int) -> TypecheckStep:
    """Handle typecheck_spec tool"""
    tool_msg = consume_tool_message(queue, tool_call_id)
    text = tool_msg.text()
    success = text.strip() == "Typecheck passed"

    return TypecheckStep(
        type="typecheck",
        vfs_snapshot=vfs_version,
        success=success,
        output=text
    )


def handle_cvl_manual_search(step: dict, queue: EventQueue, tool_call_id: str, vfs_version: int) -> SearchStep:
    """Handle cvl_manual_search tool"""
    import re

    query = step["input"]["question"]
    tool_msg = consume_tool_message(queue, tool_call_id)
    if isinstance(tool_msg.content, str):
        return SearchStep(vfs_snapshot=vfs_version, query=query, results=[], type="search")

    results: List[SearchResult] = []
    for block in tool_msg.content:
        assert isinstance(block, dict), str(block)
        title = block["title"]
        text_content = ""
        similarity = 0.0

        for cb in block["content"]:
            text_content = cb["text"]
            sim_match = re.search(r'\(Similarity:\s*([\d.]+)\)', text_content)
            if sim_match:
                similarity = float(sim_match.group(1))
                text_content = re.sub(r'\s*\(Similarity:\s*[\d.]+\)\s*$', '', text_content)

        results.append(SearchResult(
            header=title,
            content=text_content,
            similarity=similarity
        ))

    return SearchStep(
        type="search",
        vfs_snapshot=vfs_version,
        query=query,
        results=results
    )


def normalize_rejection_reason(reason: str, filename: str = "rules.spec") -> str:
    """Replace temporary file names like tmpXXXX.spec/.sol with the actual filename in error messages."""
    import re
    ext = filename.split('.')[-1] if '.' in filename else 'spec'
    return re.sub(rf'tmp[a-zA-Z0-9_]+\.{ext}', filename, reason)


def handle_put_cvl(queue: EventQueue, tool_call_id: str, vfs: VFSManager) -> PutFileStep:
    """
    Handle put_cvl tool.

    For accepted updates: consume SpecUpdatedEvent to get pretty-printed result.
    For rejected updates: show "Input not shown" (can't pretty-print invalid JSON).
    """
    tool_msg = consume_tool_message(queue, tool_call_id)
    result_text = tool_msg.text()
    accepted = result_text.strip() == "Accepted"

    path = "rules.spec"
    update: List[FileUpdate]

    if accepted:
        # Consume the SpecUpdatedEvent that follows
        spec_event = queue.take()
        assert isinstance(spec_event, SpecUpdatedEvent), f"Expected SpecUpdatedEvent after accepted put_cvl, got {type(spec_event)}"
        new_spec = spec_event.new_spec

        if path not in vfs.curr_data:
            update = [FreshFile(
                type="fresh",
                path=path,
                contents=new_spec
            )]
        else:
            curr_version = vfs.curr_data[path]
            unified = compute_diff(path, curr_version=curr_version, new_version=new_spec)
            update = [Diff(
                type="diff",
                path=path,
                contents=new_spec,
                diff_lines=unified
            )]

        vfs.push_update({path: new_spec})
    else:
        # Rejected - can't show input (invalid JSON)
        update = [FreshFile(
            type="fresh",
            path=path,
            contents="(Input not shown - invalid JSON)",
            rejected=True,
            rejection_reason=normalize_rejection_reason(result_text.strip())
        )]

    return PutFileStep(
        type="put_file",
        vfs_snapshot=vfs.curr_version,
        updates=update
    )


def handle_put_cvl_raw(step: dict, queue: EventQueue, tool_call_id: str, vfs: VFSManager) -> PutFileStep:
    """Handle put_cvl_raw tool - accepts raw CVL surface syntax"""
    tool_msg = consume_tool_message(queue, tool_call_id)
    result_text = tool_msg.text()
    accepted = result_text.strip() == "Accepted"

    cvl_content = step["input"]["cvl_file"]
    assert isinstance(cvl_content, str)

    path = "rules.spec"
    update: List[FileUpdate]

    if accepted:
        # Consume the SpecUpdatedEvent
        spec_event = queue.take()
        assert isinstance(spec_event, SpecUpdatedEvent)

        if path not in vfs.curr_data:
            update = [FreshFile(type="fresh", path=path, contents=cvl_content)]
        else:
            curr_version = vfs.curr_data[path]
            unified = compute_diff(path, curr_version=curr_version, new_version=cvl_content)
            update = [Diff(type="diff", path=path, contents=cvl_content, diff_lines=unified)]
        vfs.push_update({path: cvl_content})
    else:
        rejection = normalize_rejection_reason(result_text.strip())
        if path not in vfs.curr_data:
            update = [FreshFile(type="fresh", path=path, contents=cvl_content, rejected=True, rejection_reason=rejection)]
        else:
            curr_version = vfs.curr_data[path]
            unified = compute_diff(path, curr_version=curr_version, new_version=cvl_content)
            update = [Diff(type="diff", path=path, contents=cvl_content, diff_lines=unified, rejected=True, rejection_reason=rejection)]

    return PutFileStep(type="put_file", vfs_snapshot=vfs.curr_version, updates=update)


def handle_put_interface(step: dict, queue: EventQueue, tool_call_id: str, vfs: VFSManager) -> PutFileStep:
    """Handle put_interface tool"""
    tool_msg = consume_tool_message(queue, tool_call_id)
    result_text = tool_msg.text()
    accepted = result_text.strip() == "Accepted"

    contents = step["input"]["contents"]
    assert isinstance(contents, str)

    path = "Intf.sol"
    update: List[FileUpdate]

    if accepted:
        # Consume the IntfUpdatedEvent
        intf_event = queue.take()
        assert isinstance(intf_event, IntfUpdatedEvent)

        if path not in vfs.curr_data:
            update = [FreshFile(type="fresh", path=path, contents=contents)]
        else:
            curr_version = vfs.curr_data[path]
            unified = compute_diff(path, curr_version=curr_version, new_version=contents)
            update = [Diff(type="diff", path=path, contents=contents, diff_lines=unified)]
        vfs.push_update({path: contents})
    else:
        rejection = normalize_rejection_reason(result_text.strip(), path)
        if path not in vfs.curr_data:
            update = [FreshFile(type="fresh", path=path, contents=contents, rejected=True, rejection_reason=rejection)]
        else:
            curr_version = vfs.curr_data[path]
            unified = compute_diff(path, curr_version=curr_version, new_version=contents)
            update = [Diff(type="diff", path=path, contents=contents, diff_lines=unified, rejected=True, rejection_reason=rejection)]

    return PutFileStep(type="put_file", vfs_snapshot=vfs.curr_version, updates=update)


def handle_generation_complete(step: dict, queue: EventQueue, tool_call_id: str, vfs_version: int) -> ResultStep:
    """Handle generation_complete tool"""
    consume_tool_message(queue, tool_call_id)
    result_input = step["input"]

    return ResultStep(
        type="result",
        vfs_snapshot=vfs_version,
        expected_solc=result_input.get("expected_solc", ""),
        expected_contract_name=result_input.get("expected_contract_name", ""),
        implementation_notes=result_input.get("implementation_notes", "")
    )


def handle_get_cvl(queue: EventQueue, tool_call_id: str) -> Command:
    """Handle get_cvl tool"""
    tool_msg = consume_tool_message(queue, tool_call_id)
    return Command(type="cmd", cmd="cat rules.spec", stdout=tool_msg.text())


def handle_get_document(queue: EventQueue, tool_call_id: str) -> Command:
    """Handle get_document tool"""
    tool_msg = consume_tool_message(queue, tool_call_id)
    return Command(type="cmd", cmd="cat document.txt", stdout=tool_msg.text())


def handle_human_in_the_loop_event(step: dict, queue: EventQueue, tool_call_id: str, vfs_version: int) -> QuestionStep:
    """Handle human_in_the_loop tool"""
    tool_msg = consume_tool_message(queue, tool_call_id)
    content = tool_msg.text()
    if content.startswith("Human Response: "):
        content = content[len("Human Response: "):]

    question_input = step["input"]
    return QuestionStep(
        vfs_snapshot=vfs_version,
        type="question",
        context=question_input["context"],
        query=question_input["question"],
        answer=content.strip(),
        code=question_input.get("code", None)
    )


# ========================================
# Event Processing
# ========================================

def extract_document(events: list[ViewEvent]) -> str:
    queue = EventQueue(events)
    t = queue.take()
    assert isinstance(t, MessageEvent) and isinstance(t.message, SystemMessage)
    t = queue.take()
    assert isinstance(t, MessageEvent) and isinstance(t.message, HumanMessage)
    return cast(str, t.message.content[2])

def process_events(events: List[ViewEvent], vfs: VFSManager) -> List[Steps]:
    """Process event stream and generate Steps"""
    steps: List[Steps] = []
    queue = EventQueue(events)

    ai_messages: List[AIStepMessage] = []

    while queue.has_next():
        event = queue.peek()

        # Handle summarization events
        if isinstance(event, SummarizationEvent):
            assert not ai_messages
            queue.take()
            steps.append(SummarizationStep(
                type="summarization",
                vfs_snapshot=vfs.curr_version,
                summary_md=event.summary_text
            ))
            continue

        # Skip state update events that weren't consumed by tool handlers
        # (This can happen if state was updated without a corresponding tool call)
        if isinstance(event, (SpecUpdatedEvent, IntfUpdatedEvent)):
            if ai_messages:
                assert False
            queue.take()
            continue

        # Handle message events
        if isinstance(event, MessageEvent):
            queue.take()
            msg = event.message

            # Skip non-AI messages
            if not isinstance(msg, AIMessage):
                continue

            # Process AI message content
            cont: List[str | dict]
            if isinstance(msg.content, str):
                cont = [msg.content]
            else:
                cont = msg.content

            for content_part in cont:
                if isinstance(content_part, str):
                    if content_part.strip():
                        ai_messages.append(AIStepMessage(type="text", text=content_part))
                    continue

                ty = content_part.get("type")
                match ty:
                    case "thinking":
                        ai_messages.append({"type": "thinking", "text": content_part["thinking"]})
                    case "text":
                        if content_part.get("text", "").strip():
                            ai_messages.append({"type": "text", "text": content_part["text"]})
                    case "tool_use":
                        tool_call_id = content_part["id"]
                        tool_name = content_part["name"]

                        # Handle memory_tool by adding "edit memory" message
                        if tool_name == "memory":
                            if not ai_messages or ai_messages[-1]["type"] != "memory":
                                ai_messages.append({"type": "memory", "text": "(Memory access)"})
                            # Consume the tool message
                            nxt = queue.peek()
                            if nxt is not None and isinstance(nxt, MessageEvent):
                                nxt_msg = nxt.message
                                if isinstance(nxt_msg, ToolMessage) and nxt_msg.tool_call_id == tool_call_id:
                                    queue.take()
                            continue

                        # Create AI step for thinking/text before tool call
                        if ai_messages:
                            steps.append(AIStep(
                                vfs_snapshot=vfs.curr_version,
                                type="ai",
                                messages=ai_messages,
                                tool=tool_name
                            ))
                            ai_messages = []

                        # Handle specific tools
                        match tool_name:
                            case "cvl_manual_search":
                                steps.append(handle_cvl_manual_search(content_part, queue, tool_call_id, vfs.curr_version))

                            case "put_cvl":
                                steps.append(handle_put_cvl(queue, tool_call_id, vfs))

                            case "put_cvl_raw":
                                steps.append(handle_put_cvl_raw(content_part, queue, tool_call_id, vfs))

                            case "put_interface":
                                steps.append(handle_put_interface(content_part, queue, tool_call_id, vfs))

                            case "guidelines_judge":
                                steps.append(handle_guidelines_judge(queue, tool_call_id, vfs.curr_version))

                            case "suggestion_oracle":
                                steps.append(handle_suggestion_oracle(queue, tool_call_id, vfs.curr_version))

                            case "typecheck_spec":
                                steps.append(handle_typecheck_spec(queue, tool_call_id, vfs.curr_version))

                            case "generation_complete" | "result":
                                steps.append(handle_generation_complete(content_part, queue, tool_call_id, vfs.curr_version))

                            case "human_in_the_loop":
                                steps.append(handle_human_in_the_loop_event(content_part, queue, tool_call_id, vfs.curr_version))

                            case "get_cvl" | "get_document":
                                commands: List[VFSInteraction] = []
                                if tool_name == "get_cvl":
                                    commands.append(handle_get_cvl(queue, tool_call_id))
                                else:
                                    commands.append(handle_get_document(queue, tool_call_id))

                                steps.append(VFSStep(
                                    type="vfs",
                                    vfs_snapshot=vfs.curr_version,
                                    commands=commands
                                ))

                            case _:
                                print(f"Unhandled tool: {tool_name}")
                                # Try to consume the tool message
                                nxt = queue.peek()
                                if nxt is not None and isinstance(nxt, MessageEvent):
                                    nxt_msg = nxt.message
                                    if isinstance(nxt_msg, ToolMessage) and nxt_msg.tool_call_id == tool_call_id:
                                        queue.take()

    return steps


# ========================================
# Main Execution
# ========================================

def main():
    # Get the latest checkpoint for the thread
    latest = checkpointer.get_tuple({
        "configurable": {"thread_id": thread_id}
    })

    if latest is None:
        print(f"No checkpoint found for thread_id: {thread_id}")
        sys.exit(1)

    # Build checkpoint chain (oldest first)
    checkpoints = get_checkpoint_chain(latest)

    if not checkpoints:
        print("No checkpoints in chain")
        sys.exit(1)

    # Build unified event stream
    events = build_message_view(checkpoints)

    # Extract document from first checkpoint's messages
    document = extract_document(events)

    # Initialize VFS with document
    vfs = VFSManager({"document.txt": document})

    # Build steps list
    steps: List[Steps] = []

    # Add initial step
    steps.append(InitialStep(
        type="initial",
        vfs_snapshot=0,
        document=document
    ))

    # Process events into steps
    steps.extend(process_events(events, vfs))

    # Render template
    output = load_jinja_template(
        "spec-trace-explorer.html.j2",
        fs_dump=json.dumps(vfs.fs),
        steps_dump=json.dumps(steps)
    )

    with open(output_file, "w") as out:
        out.write(output)

    print(f"Trace written to {output_file}")


if __name__ == "__main__":
    main()
