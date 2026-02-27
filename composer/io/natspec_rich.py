from typing import Any

from textual.containers import VerticalScroll
from textual.widgets import Static, Input, Collapsible
from textual.validation import Function, Validator

from rich.syntax import Syntax
from rich.text import Text

from composer.io.ide_bridge import IDEBridge
from composer.io.tool_display import NatSpecToolDisplay
from composer.io.rich_console import BaseRichConsoleApp, _DOT

from composer.spec.ptypes import HumanQuestionSchema, NatSpecState


class NatSpecRichApp(BaseRichConsoleApp[HumanQuestionSchema, Any]):
    """Textual TUI for the NatSpec generation workflow."""

    def __init__(self, show_checkpoints: bool = False, ide: IDEBridge | None = None):
        super().__init__(
            tool_config=NatSpecToolDisplay(),
            show_checkpoints=show_checkpoints,
            ide=ide,
        )
        # Snapshot map: snap_id → (content, filename, lang)
        self._snapshots: dict[int, tuple[str, str, str | None]] = {}
        self._next_snap_id: int = 0
        # Track latest for display_result
        self._latest_spec: str | None = None
        self._latest_intf: str | None = None

    # ── Abstract method implementations ───────────────────────

    def build_interaction(self, ty: HumanQuestionSchema) -> tuple[Text, str, list[Validator]]:
        prompt = Text.assemble(
            ("QUESTION\n\n", "bold"),
            ("Question: ", "bold"), ty.question, "\n",
            ("Context: ", "bold"), ty.context,
        )
        return (prompt, "Enter your response", [])

    async def render_progress(self, target: VerticalScroll, path: list[str], upd: Any) -> None:
        pass  # NatSpec has no progress events

    # ── Snapshot helpers ──────────────────────────────────────

    def _save_snapshot(self, content: str, filename: str, lang: str | None) -> int:
        snap_id = self._next_snap_id
        self._next_snap_id += 1
        self._snapshots[snap_id] = (content, filename, lang)
        return snap_id

    # ── State extras ──────────────────────────────────────────

    async def render_state_extras(self, target: VerticalScroll, node_name: str, node_data: dict) -> None:
        if "curr_spec" in node_data:
            spec: str = node_data["curr_spec"]
            self._latest_spec = spec
            self._reset_tool_collapsing()
            if self._ide is not None:
                snap_id = self._save_snapshot(spec, "rules.spec", None)
                markup = (
                    f"[cyan]{_DOT}[/cyan]"
                    f"[@click=app.show_snapshot({snap_id})]"
                    f"[bold underline cyan]Spec updated — click to view[/bold underline cyan][/]"
                )
                await self._mount_to(target, Static(markup))
            else:
                syntax = Syntax(spec, "cvl", theme="monokai", line_numbers=True)
                coll = Collapsible(Static(syntax), title="Spec updated", collapsed=True)
                await self._mount_to(target, coll)

        if "curr_intf" in node_data:
            intf: str = node_data["curr_intf"]
            self._latest_intf = intf
            self._reset_tool_collapsing()
            if self._ide is not None:
                snap_id = self._save_snapshot(intf, "Interface.sol", "solidity")
                markup = (
                    f"[cyan]{_DOT}[/cyan]"
                    f"[@click=app.show_snapshot({snap_id})]"
                    f"[bold underline cyan]Interface updated — click to view[/bold underline cyan][/]"
                )
                await self._mount_to(target, Static(markup))
            else:
                syntax = Syntax(intf, "solidity", theme="monokai", line_numbers=True)
                coll = Collapsible(Static(syntax), title="Interface updated", collapsed=True)
                await self._mount_to(target, coll)


    # ── IDE action methods ────────────────────────────────────

    def action_show_snapshot(self, snap_id: int) -> None:
        snap = self._snapshots.get(snap_id)
        if snap is None or self._ide is None:
            return
        content, filename, lang = snap
        self.run_worker(
            self._ide_show_file(content, filename, lang),
            thread=False,
        )

    # ── Result display (NatSpecIOHandler protocol) ────────────

    async def display_result(self, final_state: NatSpecState) -> None:
        await self._mounted.wait()
        target = self.query_one("#event-log", VerticalScroll)
        assert "result" in final_state

        result = final_state["result"]

        await self._mount_to(
            target,
            Static(Text("━━ SPEC GENERATION COMPLETED ━━", style="bold green"))
        )

        # Show result metadata
        meta = Text.assemble(
            ("Contract: ", "bold"), result.expected_contract_name, "\n",
            ("Solidity version: ", "bold"), result.expected_solc, "\n",
            ("Implementation notes: ", "bold"), result.implementation_notes,
        )
        await self._mount_to(target, Static(meta))

        curr_spec = final_state.get("curr_spec") or ""
        curr_intf = final_state.get("curr_intf") or ""
        intf_filename = f"I{result.expected_contract_name}.sol"
        files = {
            "rules.spec": curr_spec,
            intf_filename: curr_intf,
        }

        if self._ide is not None:
            # Show files collapsed in TUI for reference
            spec_syntax = Syntax(curr_spec, "cvl", theme="monokai", line_numbers=True)
            await self._mount_to(
                target,
                Collapsible(Static(spec_syntax), title="rules.spec", collapsed=True),
            )
            intf_syntax = Syntax(curr_intf, "solidity", theme="monokai", line_numbers=True)
            await self._mount_to(
                target,
                Collapsible(Static(intf_syntax), title=intf_filename, collapsed=True),
            )

            # Preview in VS Code
            preview_id: str | None = None
            try:
                preview_id = await self._ide.preview_results(files)
            except Exception:
                self.notify("Failed to preview results in VS Code", severity="warning")

            if preview_id is not None:
                prompt_widget = Static(Text.assemble(
                    ("Results previewed in VS Code.\n", "bold"),
                    ("Type ACCEPT to write files or REJECT to discard.", "dim"),
                ))
                hint_widget = Static("Response must be ACCEPT or REJECT", classes="interaction-hint")
                input_widget = Input(placeholder="ACCEPT / REJECT", validate_on=["submitted"])
                input_widget.validators = [Function(
                    lambda x: x.strip().upper() in ("ACCEPT", "REJECT"),
                    "Response must be ACCEPT or REJECT",
                )]
                await self._mount_to(target, prompt_widget, input_widget, hint_widget)
                input_widget.focus()

                response = await self._input_queue.get()
                await prompt_widget.remove()
                await input_widget.remove()
                await hint_widget.remove()
                decision = response.strip().upper()

                if decision == "ACCEPT":
                    try:
                        written = await self._ide.accept_results(preview_id)
                        await self._mount_to(
                            target,
                            Static(Text(f"Results accepted — wrote {len(written)} file(s).", style="bold green"))
                        )
                    except Exception:
                        self.notify("Failed to accept results in VS Code", severity="warning")
                else:
                    try:
                        await self._ide.reject_results(preview_id)
                    except Exception:
                        pass
                    await self._mount_to(
                        target,
                        Static(Text("Results rejected.", style="yellow"))
                    )
            else:
                await self._mount_to(
                    target,
                    Static(Text("Preview unavailable — results shown above.", style="dim"))
                )
        else:
            # No IDE — show files expanded
            spec_syntax = Syntax(curr_spec, "cvl", theme="monokai", line_numbers=True)
            await self._mount_to(
                target,
                Collapsible(Static(spec_syntax), title="rules.spec", collapsed=False),
            )
            intf_syntax = Syntax(curr_intf, "solidity", theme="monokai", line_numbers=True)
            await self._mount_to(
                target,
                Collapsible(Static(intf_syntax), title=intf_filename, collapsed=False),
            )

        self._graph_done = True
        await self._mount_to(
            target,
            Static(Text("Press q to quit.", style="dim"))
        )
