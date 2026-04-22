from langchain_core.messages import ToolMessage
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ToolDisplay:
    """Declarative display config for a single tool."""

    display_name: Callable[[dict], str] | str
    """
    Label shown when the tool is called.  ``str`` for a static name, callable
    ``(input) -> str`` to vary based on the concrete arguments.
    """

    result: str | Callable[[str, ToolMessage], str | None | tuple[str, str]] | None
    """
    How to render the tool result.

    * ``None`` — suppress the result entirely.
    * ``str`` — static result label; tool message content shown as-is.
    * ``callable(name, msg)`` — dynamic.  Return ``None`` to suppress,
      ``str`` for a label (message content as body), or ``(label, body)``
      to override both.
    """


@dataclass
class GroupedTool:
    """Tool where successive calls are collapsed into a single line."""

    group_id: str
    """Identifier shared by all tools that belong to this group."""

    extract_group_items: Callable[[dict], str | list[str]]
    """From the tool arguments, extract item label(s) for the collapsed display."""

    group_display: Callable[[list[str]], str] | str
    """
    Build the collapsed display line.

    * ``str`` — rendered as ``"{group_display}: item1, item2"``.
    * ``callable(items)`` — full control.
    """

    def render_group(self, items: list[str]) -> str:
        if isinstance(self.group_display, str):
            return f"{self.group_display}: {', '.join(items)}"
        return self.group_display(items)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _suppress_ack(
    label: str,
    acks: tuple[str, ...] = ("Success", "Accepted"),
) -> Callable[[str, ToolMessage], str | None]:
    """Factory: suppress results whose content is a bare ACK.

    *acks* lists the exact strings to treat as acknowledgements.
    """
    def _check(_name: str, msg: ToolMessage) -> str | None:
        if msg.text.startswith(acks):
            return None
        return label
    return _check


def _format_cvl_result(_name: str, msg: ToolMessage) -> tuple[str, str] | None:
    """Render CVL manual search results (returned as Anthropic content blocks)."""
    raw = msg.content
    if isinstance(raw, str):
        return ("CVL Manual results", raw)
    if not isinstance(raw, list):
        return ("CVL Manual results", str(raw))
    parts: list[str] = []
    for block in raw:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        title = block.get("title", "")
        content_blocks = block.get("content", [])
        texts = []
        for cb in content_blocks:
            if isinstance(cb, dict) and cb.get("type") == "text":
                texts.append(cb.get("text", ""))
            else:
                texts.append(str(cb))
        body = "\n".join(texts)
        if title:
            parts.append(f"## {title}\n{body}")
        else:
            parts.append(body)
    if not parts:
        return None
    return ("CVL Manual results", "\n\n".join(parts))


# ---------------------------------------------------------------------------
# Reusable tool entries
# ---------------------------------------------------------------------------

class CommonTools:
    # -- Individual entries --------------------------------------------------

    cvl_manual = ToolDisplay(
        lambda p: f"Searching CVL Manual: {p.get('question', '?')[:60]}",
        _format_cvl_result,
    )
    memory = GroupedTool(
        "memory",
        lambda p: f'{p.get("command", "?")} {p.get("path", "")}'.strip(),
        lambda items: f"Accessing memory x{len(items)}",
    )
    result = ToolDisplay("Delivering result", _suppress_ack("Result"))

    code_explorer = ToolDisplay(lambda q: f"Code Exploration Request: {q["question"]}", "Code Explorer Answer")

    get_file = GroupedTool(
        "read",
        lambda p: p.get("path", "?"),
        lambda items: f"Reading: {', '.join(items)}",
    )
    put_file = GroupedTool(
        "write",
        lambda p: ", ".join(p.get("files", {}).keys()),
        lambda items: f"Wrote: {', '.join(items)}",
    )
    list_files = ToolDisplay("Listing files", "File listing")
    grep_files = ToolDisplay(
        lambda p: f"Searching files for: {p.get('search_string', '?')}",
        "Search results",
    )
    write_rough_draft = ToolDisplay("Write rough draft", None)
    read_rough_draft = ToolDisplay("Read rough draft", "Rough Draft")
    extended_reasoning = ToolDisplay("Reasoning", None)
    cvl_keyword_search = ToolDisplay(
        lambda p: f"CVL Manual Search: {p.get('query')}", "CVL Matching Sections",
    )
    get_cvl_manual_section = ToolDisplay(
        lambda p: f"Read CVL Manual: {' / '.join(p.get('headers', []))}", None,
    )
    cvl_research = ToolDisplay(
        lambda p: f"Researching CVL: {p.get('question', '?')}", "Research result",
    )
    scan_knowledge_base = ToolDisplay("Scanning knowledge base", "KB scan results")
    get_knowledge_base_article = ToolDisplay("Reading KB article", "KB article")
    knowledge_base_contribute = ToolDisplay("Contributing to KB", "KB contribution")

    # -- Grouped display bundles ---------------------------------------------
    # Each corresponds to a capability provider (builder / service).
    # Use **CommonTools.source_displays() etc. when composing a phase config.

    @staticmethod
    def source_displays() -> dict[str, "ToolDisplay | GroupedTool"]:
        """Display entries for tools from ``fs_tools()`` (SourceBuilder)."""
        return {
            "get_file": CommonTools.get_file,
            "put_file": CommonTools.put_file,
            "list_files": CommonTools.list_files,
            "grep_files": CommonTools.grep_files,
            "explore_code": CommonTools.code_explorer
        }

    @staticmethod
    def cvl_manual_displays() -> dict[str, "ToolDisplay | GroupedTool"]:
        """Display entries for tools from ``cvl_manual_tools()`` (CVLOnlyBuilder)."""
        return {
            "cvl_manual_search": CommonTools.cvl_manual,
            "cvl_keyword_search": CommonTools.cvl_keyword_search,
            "get_cvl_manual_section": CommonTools.get_cvl_manual_section,
        }

    @staticmethod
    def kb_displays() -> dict[str, "ToolDisplay"]:
        """Display entries for tools from ``kb_tools()`` (WorkflowServices)."""
        return {
            "scan_knowledge_base": CommonTools.scan_knowledge_base,
            "get_knowledge_base_article": CommonTools.get_knowledge_base_article,
            "knowledge_base_contribute": CommonTools.knowledge_base_contribute,
        }

    @staticmethod
    def rough_draft_displays() -> dict[str, "ToolDisplay"]:
        """Display entries for rough draft tools."""
        return {
            "write_rough_draft": CommonTools.write_rough_draft,
            "read_rough_draft": CommonTools.read_rough_draft,
        }

    @staticmethod
    def cvl_research_displays() -> dict[str, "ToolDisplay | GroupedTool"]:
        """Display entries for the CVL research sub-agent and all tools it uses."""
        return {
            "cvl_research": CommonTools.cvl_research,
            **CommonTools.cvl_manual_displays(),
            **CommonTools.kb_displays(),
            **CommonTools.rough_draft_displays(),
        }
    
    @staticmethod
    def feedback_tools() -> dict[str, "ToolDisplay | GroupedTool"]:
        return {
            "feedback_tool": ToolDisplay("Getting feedback", "Feedback"),
            "record_skip": ToolDisplay(
                lambda p: f"Skipping property #{p.get('property_index', '?')}",
                _suppress_ack("Skip result", ("Recorded skip",)),
            ),
            "unskip_property": ToolDisplay(
                lambda p: f"Un-skipping property #{p.get('property_index', '?')}",
                _suppress_ack("Unskip result", ("Removed skip",)),
            ),
        }
    
    @staticmethod
    def cvl_manipulation() -> dict[str, ToolDisplay]:
        return {
            "put_cvl": ToolDisplay("Writing spec", _suppress_ack("Spec write result")),
            "put_cvl_raw": ToolDisplay("Writing spec", _suppress_ack("Spec write result")),
            "get_cvl": ToolDisplay("Reading spec", None),
        }


# ---------------------------------------------------------------------------
# Config wrapper
# ---------------------------------------------------------------------------

@dataclass
class ToolDisplayConfig:
    """Declarative mapping from tool names to display rules."""

    tool_display: dict[str, ToolDisplay | GroupedTool] = field(default_factory=dict)

    # -- tool call formatting ------------------------------------------------

    def format_tool_call(self, name: str, input: dict) -> str:
        """Return a user-friendly label for a tool invocation."""
        entry = self.tool_display.get(name)
        if entry is None or isinstance(entry, GroupedTool):
            return f"Tool: {name}"
        nm = entry.display_name
        if isinstance(nm, str):
            return nm
        return nm(input)

    # -- grouping ------------------------------------------------------------

    def get_group(self, name: str) -> GroupedTool | None:
        """Return the ``GroupedTool`` entry for *name*, or ``None``."""
        entry = self.tool_display.get(name)
        return entry if isinstance(entry, GroupedTool) else None

    # -- result formatting ---------------------------------------------------

    def format_result(self, name: str, msg: ToolMessage) -> tuple[str, str] | None:
        """Format a tool result for display.

        Returns ``(label, body)`` for the collapsible, or ``None`` to suppress.
        """
        entry = self.tool_display.get(name)
        content = msg.text()

        if isinstance(entry, GroupedTool):
            return None

        if entry is None:
            return (name, content)

        r = entry.result
        if r is None:
            return None
        if callable(r):
            out = r(name, msg)
            if out is None:
                return None
            if isinstance(out, tuple):
                return out
            return (out, content)
        return (r, content)


# ---------------------------------------------------------------------------
# Concrete configs
# ---------------------------------------------------------------------------

class CodeGenToolDisplay(ToolDisplayConfig):
    """Tool display configuration for the code generation workflow."""

    def __init__(self):
        super().__init__(tool_display={
            "certora_prover": ToolDisplay(
                lambda p: (
                    "Running prover: " + p.get("target_contract", "")
                    + (f" — rule {p['rule']}" if p.get("rule") else "")
                ),
                None,
            ),
            
            "put_file": CommonTools.put_file,
            "get_file": CommonTools.get_file,
            "list_files": CommonTools.list_files,
            "grep_files": CommonTools.grep_files,
            "propose_spec_change": ToolDisplay(
                lambda p: (
                    f"Proposing spec change: {p['explanation']}"
                    if p.get("explanation") else "Proposing spec change"
                ),
                None,
            ),
            "human_in_the_loop": ToolDisplay(
                lambda p: (
                    f"Asking for input: {p['question']}"
                    if p.get("question") else "Asking for input"
                ),
                None,
            ),
            "code_result": ToolDisplay("Finalizing result", _suppress_ack("Final result")),
            "cvl_manual_search": CommonTools.cvl_manual,
            "requirements_evaluation": ToolDisplay(
                "Evaluating requirements", "Requirements evaluation",
            ),
            "requirement_relaxation_request": ToolDisplay(
                lambda p: (
                    f"Requesting requirement relaxation #{p.get('req_number', '?')}: {p.get('req_text', '')}"
                    if p.get("req_text")
                    else "Requesting requirement relaxation"
                ),
                None,
            ),
            "write_rough_draft": CommonTools.write_rough_draft,
            "read_rough_draft": CommonTools.read_rough_draft,
            "extended_reasoning": CommonTools.extended_reasoning,
            "memory": CommonTools.memory,
        })


class NatSpecToolDisplay(ToolDisplayConfig):
    """Tool display configuration for the NatSpec generation workflow."""

    def __init__(self):
        super().__init__(tool_display={
            "cvl_manual_search": CommonTools.cvl_manual,
            "human_in_the_loop": ToolDisplay(
                lambda p: (
                    f"Asking for input: {p['question']}"
                    if p.get("question") else "Asking for input"
                ),
                None,
            ),
            "result": CommonTools.result,
            "put_interface": ToolDisplay("Writing interface", _suppress_ack("Interface write result")),
            "get_document": ToolDisplay("Reading design doc", "Design document"),
            "get_cvl": ToolDisplay("Reading spec", None),
            "put_cvl_raw": ToolDisplay("Writing spec", _suppress_ack("Spec write result")),
            "put_cvl": ToolDisplay("Writing spec", _suppress_ack("Spec write result")),
            "guidelines_judge": ToolDisplay("Running guidelines judge", "Guidelines feedback"),
            "suggestion_oracle": ToolDisplay("Getting suggestions", "Suggestions"),
            "typecheck_spec": ToolDisplay("Type-checking spec", "Type-check result"),
            "memory": CommonTools.memory,
        })
