from typing import Callable

import difflib
import pathlib

from rich.console import Console

from composer.diagnostics.handlers import summarize_update, print_prover_updates
from composer.diagnostics.stream import ProgressUpdate
from composer.human.types import HumanInteractionType, ProposalType, QuestionType, RequirementRelaxationType, ExtractionQuestionType
from composer.ui.prompt import prompt_input
from composer.io.protocol import WorkflowPurpose
from composer.core.state import ResultStateSchema, AIComposerState


from graphcore.tools.vfs import VFSAccessor


class BaseConsoleHandler[H, P]:
    """Common console-based IOHandler functionality shared across workflows."""

    async def log_checkpoint_id(self, *, path: list[str], checkpoint_id: str):
        print("current checkpoint: " + checkpoint_id)

    async def log_start(self, *, path: list[str], description: str, tool_id: str | None):
        print(f"[{description}]")

    async def log_end(self, path: list[str]):
        if len(path) > 1:
            print(f"[Nested workflow end] {' > '.join(path)}")
        else:
            print(f"[Workflow end] {path[0]}")

    def _print_header(self, topic: str) -> None:
        print("\n" + "=" * 80)
        print(topic)
        print("=" * 80)


class ConsoleHandler(BaseConsoleHandler[HumanInteractionType, ProgressUpdate]):
    def __init__(
        self,
        capture_prover_output: bool = False,
        output_folder: str | None = None,
    ):
        self._capture_prover_output = capture_prover_output
        # Pre-resolved by the CLI entry point: ``--output-folder`` if
        # explicitly given, else ``--source-root`` if set, else ``None``.
        # ``None`` triggers an interactive prompt at output time;
        # blank input there skips the disk write.
        self._output_folder = output_folder

    async def log_workflow_thread(self, purpose: WorkflowPurpose, thread_id: str) -> None:
        print(f"[{purpose.value}] thread: {thread_id}")

    async def show_error(self, error: Exception) -> None:
        import traceback
        self._print_header("WORKFLOW ERROR")
        print(f"{type(error).__name__}: {error}\n")
        traceback.print_exception(error)

    async def log_state_update(self, path: list[str], st: dict):
        summarize_update(st)

    async def progress_update(self, path: list[str], upd: ProgressUpdate):
        if self._capture_prover_output and upd["type"] in ("prover_output", "cloud_polling"):
            return
        print_prover_updates(upd)

    def handle_proposal_interrupt(self, interrupt_ty: ProposalType, debug_thunk: Callable[[], None]) -> str:
        self._print_header("SPEC CHANGE PROPOSAL")
        orig = interrupt_ty["current_spec"].splitlines(keepends=True)
        proposed = interrupt_ty["proposed_spec"].splitlines(keepends=True)

        diff = difflib.unified_diff(
            a = orig,
            fromfile="a/rules.spec",
            b = proposed,
            tofile="b/rules.spec",
            n=3,
        )

        print(f"Explanation: {interrupt_ty['explanation']}")
        print("Proposed diff is as follows:")

        console = Console(highlighter=None, markup=False)

        for line in diff:
            if line.startswith("---"):
                console.print(line, style="bold white", end="")
            elif line.startswith("+++"):
                console.print(line, style="bold white", end="")
            elif line.startswith("@@"):
                console.print(line, style="cyan", end="")
            elif line.startswith("+"):
                console.print(line, style="green", end="")
            elif line.startswith("-"):
                console.print(line, style="red", end="")
            else:
                console.print(line, end="")

        print("")

        def filt(x: str) -> str | None:
            if not (x.startswith("ACCEPTED") or x.startswith("REJECTED") or x.startswith("REFINE")):
                return "Response must begin with ACCEPTED/REJECTED/REFINE"
            return None

        return prompt_input("Response to proposal, must start with ACCEPTED/REJECTED/REFINE", debug_thunk, filt)

    def handle_question_interrupt(self, interrupt_data: QuestionType, debug_thunk: Callable[[], None]) -> str:
        self._print_header("HUMAN ASSISTANCE REQUESTED")
        print(f"Question: {interrupt_data['question']}")
        print(f"Context: {interrupt_data['context']}")
        if interrupt_data["code"]:
            print(f"Code:\n{interrupt_data['code']}")
        return prompt_input("Enter your answer (begin response with FOLLOWUP to request clarification)", debug_thunk)

    def handle_req_relaxation_interrupt(self, interrupt: RequirementRelaxationType, debug_thunk: Callable[[], None]) -> str:
        self._print_header("REQUIREMENTS SKIP REQUEST")
        print("The agent would like to skip satisfying one of the requirements")
        print(f"Context:\n{interrupt['context']}")
        print(f"Req #{interrupt['req_number']}: {interrupt['req_text']}")
        print(f"Judgment from oracle:\n{interrupt['judgment']}")
        print(f"Explanation for request:\n{interrupt['explanation']}")
        def filt(r: str) -> str | None:
            if not r.startswith("ACCEPTED") and not r.startswith("REJECTED"):
                return "Response must begin with ACCEPTED/REJECTED"
            return None
        return prompt_input("Response to request, must start with ACCEPTED/REJECTED", debug_thunk, filt)

    def handle_extraction_question(self, interrupt: ExtractionQuestionType, debug_thunk: Callable[[], None]) -> str:
        self._print_header("HUMAN ASSISTANCE REQUESTED")
        print(f"Context:\n{interrupt['context']}")
        print(f"Question: {interrupt['question']}")
        return prompt_input("Enter your response", debug_thunk)

    async def human_interaction(
        self,
        ty: HumanInteractionType,
        debug_thunk: Callable[[], None]
    ) -> str:
        match ty["type"]:
            case "proposal":
                return self.handle_proposal_interrupt(ty, debug_thunk)
            case "question":
                return self.handle_question_interrupt(ty, debug_thunk)
            case "req_relaxation":
                return self.handle_req_relaxation_interrupt(ty, debug_thunk)
            case "extraction_question":
                return self.handle_extraction_question(ty, debug_thunk)

    async def output(
        self,
        res: ResultStateSchema,
        mat: VFSAccessor[AIComposerState],
        st: AIComposerState
    ):
        print("\n" + "=" * 80)
        print("CODE GENERATION COMPLETED")
        print("=" * 80)
        print("Generated Source Files:")
        for path in res.source:
            print(f"\n--- {path} ---")
            file_contents = mat.get(st, path)
            assert file_contents is not None
            print(file_contents.decode("utf-8"))

        print(f"\nComments: {res.comments}")

        target = self._resolve_output_folder()
        if target is None:
            print("\nNo output folder configured; skipping disk write.")
            return
        # Iterate the VFS dirty layer rather than ``res.source``: the
        # layer holds everything that needs to be persisted (agent-
        # generated sources plus mutated/new specs). The upload step
        # drops specs whose contents already match the source_root
        # underlay (see ``executor.get_fresh_input``), so iterating
        # ``state["vfs"]`` writes exactly the diff that's not already
        # on disk.
        self._write_sources(target, list(st["vfs"].items()))

    def _resolve_output_folder(self) -> pathlib.Path | None:
        """Return the target directory for the disk write, or ``None``
        when the user opts out at the prompt.

        Constructor argument wins; otherwise we ask interactively. Blank
        input at the prompt means "don't write anything".
        """
        if self._output_folder is not None:
            return pathlib.Path(self._output_folder).expanduser().resolve()
        raw = input(
            "Enter a directory to write the generated files to "
            "(blank to skip): "
        ).strip()
        if not raw:
            return None
        return pathlib.Path(raw).expanduser().resolve()

    def _write_sources(
        self, target: pathlib.Path, sources: list[tuple[str, str]],
    ) -> None:
        """Write each ``(vfs_path, content)`` pair to ``target / vfs_path``."""
        target.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting {len(sources)} file(s) to {target}:")
        for vfs_path, content in sources:
            dest = target / vfs_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)
            print(f"  {dest}")
