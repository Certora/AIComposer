from typing import Callable, Optional, cast
import difflib

from rich.console import Console

from verisafe.human.types import ProposalType, QuestionType, HumanInteractionType, RequirementRelaxationType

def prompt_input(prompt_str: str, debug_thunk: Callable[[], None], filter: Optional[Callable[[str], Optional[str]]] = None) -> str:
    l = input(prompt_str + " (double newlines ends): ")
    buffer = ""
    num_consecutive_blank = 0
    while True:
        x = l.strip()
        buffer += x + "\n"
        if x == "":
            num_consecutive_blank += 1
        else:
            num_consecutive_blank = 0
        if num_consecutive_blank == 2:
            break
        l = input("> ")
    if buffer.strip() == "DEBUG":
        debug_thunk()
        return prompt_input(prompt_str, debug_thunk, filter)
    if filter is None:
        return buffer
    filter_res = filter(buffer)
    if filter_res is None:
        return buffer
    return prompt_input(prompt_str, debug_thunk, filter)

def _print_header(topic: str) -> None:
    print("\n" + "=" * 80)
    print(topic)
    print("=" * 80)

def handle_proposal_interrupt(interrupt_ty: ProposalType, debug_thunk: Callable[[], None]) -> str:
    _print_header("SPEC CHANGE PROPOSAL")
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

    console = Console(highlighter=None)

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

    def filt(x: str) -> Optional[str]:
        if not (x.startswith("ACCEPTED") or x.startswith("REJECTED") or x.startswith("REFINE")):
            return "Response must begin with ACCEPTED/REJECTED/REFINE"
        return None

    return prompt_input("Response to proposal, must start with ACCEPTED/REJECTED/REFINE", debug_thunk, filt)

def handle_question_interrupt(interrupt_data: QuestionType, debug_thunk: Callable[[], None]) -> str:
    _print_header("HUMAN ASSISTANCE REQUESTED")
    print(f"Question: {interrupt_data['question']}")
    print(f"Context: {interrupt_data['context']}")
    if interrupt_data["code"]:
        print(f"Code:\n{interrupt_data['code']}")
    return prompt_input("Enter your answer (begin response with FOLLOWUP to request clarification)", debug_thunk)

def handle_req_relaxation_interrupt(interrupt: RequirementRelaxationType, debug_thunk: Callable[[], None]) -> str:
    _print_header("REQUIREMENTS SKIP REQUEST")
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


def handle_human_interrupt(interrupt_data: dict, debug_thunk: Callable[[], None]) -> str:
    """Handle human-in-the-loop interrupts and get user input."""
    interrupt_ty = cast(HumanInteractionType, interrupt_data)

    match interrupt_ty["type"]:
        case "proposal":
            return handle_proposal_interrupt(interrupt_ty, debug_thunk)
        case "question":
            return handle_question_interrupt(interrupt_ty, debug_thunk)
        case "req_relaxation":
            return handle_req_relaxation_interrupt(interrupt_ty, debug_thunk)
