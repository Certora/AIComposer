from dataclasses import dataclass, field


@dataclass
class ToolDisplayConfig:
    """Base configuration for how tool calls and results are displayed in the TUI."""
    tool_display: dict[str, str] = field(default_factory=dict)
    tool_result_display: dict[str, str] = field(default_factory=dict)
    collapse_groups: dict[str, str] = field(default_factory=dict)
    suppress_results: set[str] = field(default_factory=set)

    def format_tool_call(self, name: str, input: dict) -> str:
        """Return a user-friendly description of a tool call."""
        return self.tool_display.get(name, f"Tool: {name}")

    def collapse_detail(self, name: str, input: dict) -> str:
        """Extract the detail item for collapsing (e.g. file path)."""
        return ""

    def render_collapsed_text(self, group: str, items: list[str]) -> str:
        """Build the display text for a collapsed group of tool calls."""
        return f"Tools: {', '.join(items)}"

    def should_show_result(self, name: str, content: str) -> bool:
        """Return whether a tool result should be displayed. Override to add content-based filtering."""
        return name not in self.suppress_results


class CodeGenToolDisplay(ToolDisplayConfig):
    """Tool display configuration for the code generation workflow."""

    def __init__(self):
        super().__init__(
            tool_display={
                "certora_prover": "Running prover",
                "put_file": "Writing files",
                "get_file": "Reading file",
                "list_files": "Listing files",
                "grep_files": "Searching files",
                "propose_spec_change": "Proposing spec change",
                "human_in_the_loop": "Asking for input",
                "code_result": "Finalizing result",
                "cvl_manual_search": "Searching CVL manual",
                "requirements_evaluation": "Evaluating requirements",
                "requirement_relaxation_request": "Requesting requirement relaxation",
                "memory": "Accessing memory",
            },
            tool_result_display={
                "certora_prover": "Prover results",
                "put_file": "File write result",
                "get_file": "File contents",
                "list_files": "File listing",
                "grep_files": "Search results",
                "cvl_manual_search": "Manual search results",
                "requirements_evaluation": "Requirements evaluation",
                "requirement_relaxation_request": "Relaxation result",
                "propose_spec_change": "Spec change result",
                "human_in_the_loop": "Human response",
                "code_result": "Final result",
                "memory": "Memory result",
            },
            collapse_groups={
                "get_file": "read",
                "put_file": "write",
                "memory": "memory",
            },
            suppress_results={
                "human_in_the_loop",
                "propose_spec_change",
                "requirement_relaxation_request",
                "certora_prover",
            },
        )

    def format_tool_call(self, name: str, input: dict) -> str:
        base = self.tool_display.get(name, f"Tool: {name}")
        match name:
            case "certora_prover":
                target = input.get("target_contract", "")
                rule = input.get("rule")
                detail = f": {target}" + (f" — rule {rule}" if rule else "")
                return base + detail
            case "get_file":
                return f"{base}: {input.get('path', '?')}"
            case "grep_files":
                return f"{base} for: {input.get('search_string', '?')}"
            case "cvl_manual_search":
                q = input.get("question", "?")[:60]
                return f"{base}: {q}"
            case "put_file":
                files = input.get("files", {})
                return f"{base}: {', '.join(files.keys())}"
            case "memory":
                cmd = input.get("command", "?")
                path = input.get("path", "")
                return f"{base}: {cmd} {path}".strip()
            case "human_in_the_loop":
                q = input.get("question", "")
                return f"{base}: {q}" if q else base
            case "propose_spec_change":
                expl = input.get("explanation", "")
                return f"{base}: {expl}" if expl else base
            case "requirement_relaxation_request":
                num = input.get("req_number", "?")
                req = input.get("req_text", "")
                return f"{base} #{num}: {req}" if req else base
            case _:
                return base

    def collapse_detail(self, name: str, input: dict) -> str:
        match name:
            case "get_file":
                return input.get("path", "?")
            case "put_file":
                files = input.get("files", {})
                return ", ".join(files.keys())
            case _:
                return ""

    def render_collapsed_text(self, group: str, items: list[str]) -> str:
        match group:
            case "read":
                return f"Reading: {', '.join(items)}"
            case "write":
                return f"Wrote: {', '.join(items)}"
            case "memory":
                count = len(items)
                if count == 1:
                    return "Accessing memory"
                return f"Accessing memory (×{count})"
            case _:
                return f"Tools: {', '.join(items)}"


class NatSpecToolDisplay(ToolDisplayConfig):
    """Tool display configuration for the NatSpec generation workflow."""

    def __init__(self):
        super().__init__(
            tool_display={
                "cvl_manual_search": "Searching CVL manual",
                "human_in_the_loop": "Asking for input",
                "result": "Finalizing result",
                "put_interface": "Writing interface",
                "get_document": "Reading design doc",
                "get_cvl": "Reading spec",
                "put_cvl_raw": "Writing spec",
                "put_cvl": "Writing spec",
                "guidelines_judge": "Running guidelines judge",
                "suggestion_oracle": "Getting suggestions",
                "typecheck_spec": "Type-checking spec",
                "memory": "Accessing memory",
            },
            tool_result_display={
                "cvl_manual_search": "Manual search results",
                "guidelines_judge": "Guidelines feedback",
                "suggestion_oracle": "Suggestions",
                "typecheck_spec": "Type-check result",
                "result": "Generation result",
                "get_document": "Design document",
                "put_cvl": "Spec write result",
                "put_cvl_raw": "Spec write result",
                "put_interface": "Interface write result",
            },
            collapse_groups={
                "memory": "memory",
            },
            suppress_results={
                "human_in_the_loop",
                "get_cvl",
                "memory",
            },
        )

    def format_tool_call(self, name: str, input: dict) -> str:
        base = self.tool_display.get(name, f"Tool: {name}")
        match name:
            case "cvl_manual_search":
                q = input.get("question", "?")[:60]
                return f"{base}: {q}"
            case "memory":
                cmd = input.get("command", "?")
                path = input.get("path", "")
                return f"{base}: {cmd} {path}".strip()
            case "human_in_the_loop":
                q = input.get("question", "")
                return f"{base}: {q}" if q else base
            case _:
                return base

    def should_show_result(self, name: str, content: str) -> bool:
        if not super().should_show_result(name, content):
            return False
        if name in ("put_cvl", "put_cvl_raw", "put_interface") and content == "Accepted":
            return False
        return True
