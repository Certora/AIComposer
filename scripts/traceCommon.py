"""
Common utilities and type definitions for trace dump scripts.
Shared between traceDump.py (composer workflow) and specTraceDump.py (natspec workflow).
"""

import difflib
from typing import Dict, List, TypedDict, Literal, Annotated, Union, TypeVar, Generic, NotRequired, cast
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from pydantic import Discriminator


# ========================================
# Common Type Definitions
# ========================================

StepTy = TypeVar("StepTy")


class AbstractStep(TypedDict, Generic[StepTy]):
    """Base class for all step types"""
    vfs_snapshot: int
    type: StepTy


class AIStepMessage(TypedDict):
    """Message within an AI step (thinking, text, or memory edit)"""
    type: Literal["thinking", "text", "memory"]
    text: str


class AIStep(AbstractStep[Literal["ai"]]):
    """AI execution step with messages and tool call"""
    messages: List[AIStepMessage]
    tool: str


class QuestionStep(AbstractStep[Literal["question"]]):
    """Human-in-the-loop question"""
    query: str
    answer: str
    context: str
    code: NotRequired[str]


class FreshFile(TypedDict):
    """New file being created"""
    type: Literal["fresh"]
    contents: str
    path: str
    rejected: NotRequired[bool]
    rejection_reason: NotRequired[str]


class Diff(TypedDict):
    """File update with diff"""
    type: Literal["diff"]
    path: str
    contents: str
    diff_lines: List[str]
    rejected: NotRequired[bool]
    rejection_reason: NotRequired[str]


FileUpdate = Annotated[Union[Diff, FreshFile], Discriminator("type")]


class PutFileStep(AbstractStep[Literal["put_file"]]):
    """File update step"""
    updates: List[FileUpdate]


class SummarizationStep(AbstractStep[Literal["summarization"]]):
    """Context summarization event"""
    summary_md: str


class Thought(TypedDict):
    """VFS interaction - thought/reasoning"""
    type: Literal["thought"]
    msg: str


class Command(TypedDict):
    """VFS interaction - command execution"""
    type: Literal["cmd"]
    cmd: str
    stdout: str


VFSInteraction = Annotated[Thought | Command, Discriminator("type")]


class VFSStep(AbstractStep[Literal["vfs"]]):
    """VFS read operations step"""
    commands: list[VFSInteraction]


# ========================================
# VFS Manager
# ========================================

class VFSManager:
    """Manages VFS snapshots across execution"""

    def __init__(self, ver_0: Dict[str, str]):
        self.fs = [ver_0]
        self.curr_data = ver_0.copy()

    def push_update(self, upd: Dict[str, str]):
        """Add a new VFS snapshot"""
        self.fs.append(upd.copy())
        for (k, v) in upd.items():
            self.curr_data[k] = v

    @property
    def curr_version(self) -> int:
        """Get current VFS version number"""
        return len(self.fs) - 1


# ========================================
# Utility Functions
# ========================================

def compute_diff(path: str, curr_version: str, new_version: str) -> List[str]:
    """Compute unified diff between two file versions"""
    ud = difflib.unified_diff(
        a=curr_version.splitlines(keepends=True),
        b=new_version.splitlines(keepends=True),
        fromfile="a/" + path,
        tofile="b/" + path
    )
    return list(ud)


class MessageQueue:
    """Sequential message processor for checkpoint messages"""

    def __init__(self, msgs: list[BaseMessage]):
        self.msgs = msgs
        self.i = 0

    def peek(self) -> BaseMessage | None:
        """Look at next message without consuming it"""
        if self.i >= len(self.msgs):
            return None
        return self.msgs[self.i]

    def take(self) -> BaseMessage:
        """Consume and return next message"""
        to_ret = self.msgs[self.i]
        self.i += 1
        return to_ret

    def has_next(self) -> bool:
        """Check if more messages available"""
        return self.i < len(self.msgs)


# ========================================
# Common Tool Handlers
# ========================================

def has_vfs_tools(m: AIMessage) -> bool:
    """Check if AIMessage contains VFS tool calls"""
    for t in m.tool_calls:
        nm = t["name"]
        if nm == "list_files" or nm == "grep_files" or nm == "get_file":
            return True
    return False


def handle_vfs_tools(step: dict, message_queue: MessageQueue) -> list[VFSInteraction]:
    """Handle VFS tool calls (list_files, grep_files, get_file)"""
    commands: list[VFSInteraction] = []
    match step["name"]:
        case "list_files":
            nxt = message_queue.take()
            assert isinstance(nxt, ToolMessage)
            commands.append(Command(
                type="cmd",
                cmd="ls",
                stdout=nxt.text()
            ))
        case "grep_files":
            nxt = message_queue.take()
            assert isinstance(nxt, ToolMessage)
            query = step["input"]["search_string"]
            commands.append(Command(
                type="cmd",
                cmd=f"grep {query}",
                stdout=nxt.text()
            ))
        case "get_file":
            which = step["input"]["path"]
            nxt = message_queue.take()
            assert isinstance(nxt, ToolMessage)
            commands.append(Command(
                type="cmd",
                cmd=f"cat {which}",
                stdout=nxt.text()
            ))
    rem_nxt = message_queue.peek()
    if rem_nxt is not None and isinstance(rem_nxt, AIMessage) and has_vfs_tools(rem_nxt):
        commands.extend(handle_next_vfs_tool(cast(AIMessage, message_queue.take()), message_queue))
    return commands


def handle_next_vfs_tool(m: AIMessage, message_queue: MessageQueue) -> list[VFSInteraction]:
    """Process next VFS tool in chain, collecting thoughts along the way"""
    cont: List[dict | str]
    if isinstance(m.content, list):
        cont = m.content
    else:
        cont = [m.content]
    thoughts: list[str] = []
    to_ret: list[VFSInteraction] = []
    for c in cont:
        if isinstance(c, str):
            thoughts.append(c)
            continue
        ty = c.get("type")
        if ty == "text":
            thoughts.append(cast(str, c.get("text")))
            continue
        elif ty == "thinking":
            thoughts.append(cast(str, c.get("thinking")))

        # yolo
        if ty != "tool_use":
            continue
        thoughts_san = [d.strip() for d in thoughts if d.strip()]
        if len(thoughts_san) > 0:
            to_ret.append(Thought(
                type="thought",
                msg="\n".join(thoughts_san)
            ))
        to_ret.extend(handle_vfs_tools(c, message_queue))
        return to_ret
    raise RuntimeError("Didn't actually hit a tool call")


def extract_human_response(msg_queue: MessageQueue) -> str:
    """Extract human response from tool message"""
    answer_msg = msg_queue.take()
    assert isinstance(answer_msg, ToolMessage)
    return answer_msg.text()


def handle_human_in_the_loop(step: dict, msg_queue: MessageQueue, vfs_version: int) -> QuestionStep:
    """Handle human_in_the_loop tool (generic across workflows)"""
    question_input = step["input"]
    content = extract_human_response(msg_queue)
    assert isinstance(content, str)
    if content.startswith("Human Response: "):
        content = content[len("Human Response: "):]
    return QuestionStep(
        vfs_snapshot=vfs_version,
        type="question",
        context=question_input["context"],
        query=question_input["question"],
        answer=content.strip(),
        code=question_input.get("code", None)
    )
