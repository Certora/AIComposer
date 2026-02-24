from typing import Callable

import difflib

from rich.console import Console

from composer.diagnostics.handlers import summarize_update, print_prover_updates
from composer.diagnostics.stream import ProgressUpdate
from composer.human.types import HumanInteractionType, ProposalType, QuestionType, RequirementRelaxationType, ExtractionQuestionType
from composer.io.prompt import prompt_input
from composer.core.state import ResultStateSchema, AIComposerState


from graphcore.tools.vfs import VFSAccessor

class ConsoleHandler:
    async def log_thread_id(self, tid: str, chosen: bool):
        if chosen:
            print(f"Selected thread id: {tid}")

    async def log_checkpoint_id(self, *, path: list[str], checkpoint_id: str):
        print("current checkpoint: " + checkpoint_id)

    async def log_state_update(self, path: list[str], st: dict):
        summarize_update(st)

    async def progress_update(self, path: list[str], upd: ProgressUpdate):
        print_prover_updates(upd)

    async def log_start(self, *, path: list[str], tool_id: str | None):
        if len(path) > 1:
            tool_info = f" (tool={tool_id})" if tool_id else ""
            print(f"[Nested workflow start] {' > '.join(path)}{tool_info}")
        else:
            print(f"[Workflow start] {path[0]}")

    async def log_end(self, path: list[str]):
        if len(path) > 1:
            print(f"[Nested workflow end] {' > '.join(path)}")
        else:
            print(f"[Workflow end] {path[0]}")

    def _print_header(self, topic: str) -> None:
        print("\n" + "=" * 80)
        print(topic)
        print("=" * 80)

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
            content = file_contents.decode("utf-8")
            print(content)

        print(f"\nComments: {res.comments}")
