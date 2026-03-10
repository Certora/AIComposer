import pathlib
import sys

if __name__ != "__main__":
    raise RuntimeError("This is a script only module")

import bind as _

import psycopg
import json

from typing import Dict, Optional, List, cast, TypedDict, Literal, Annotated, Union, TypeVar, Generic, NotRequired
from langgraph.checkpoint.base import CheckpointTuple
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage
from pydantic import Discriminator

from composer.audit.types import ManualResult, RuleResult
from composer.audit.db import AuditDB
from composer.workflow.factories import get_checkpointer
from composer.templates.loader import load_jinja_template
from composer.natreq.judge import ClassificationType

# Import common utilities and base types
from traceCommon import (
    AbstractStep, AIStepMessage, AIStep, QuestionStep, FreshFile, Diff, FileUpdate,
    PutFileStep, SummarizationStep, Thought, Command, VFSInteraction, VFSStep,
    VFSManager, compute_diff, MessageQueue,
    has_vfs_tools, handle_vfs_tools, handle_next_vfs_tool, extract_human_response
)


# ========================================
# Composer-Specific Type Definitions
# ========================================

class InitialStep(AbstractStep[Literal["initial"]]):
    spec: str
    interface: str
    system_doc: str
    reqs: NotRequired[list[str]]

class ProverStep(AbstractStep[Literal["prover"]]):
    contract_file: str
    rule: Optional[str]
    todo_list: Optional[str]
    results: List[RuleResult]

class SearchStep(AbstractStep[Literal["search"]]):
    query: str
    results: List[ManualResult]


class ResultStep(AbstractStep[Literal["result"]]):
    comments: str
    files: List[str]

class ProposalStep(AbstractStep[Literal["proposal"]]):
    explanation: str
    human_response: str
    proposed_diff: List[str]


class RelaxationStep(AbstractStep[Literal["relaxation"]]):
    requirement: str
    judgment: ClassificationType
    explanation: str
    context: str
    response: str

class ReqEval(TypedDict):
    require_text: str
    judgment: ClassificationType | Literal["IGNORED"]
    commentary: NotRequired[str]

class RequirementsStep(AbstractStep[Literal["judge"]]):
    evaluation: list[ReqEval]

Steps = Annotated[Union[
    AIStep,
    InitialStep,
    ProverStep,
    SearchStep,
    PutFileStep,
    QuestionStep,
    ProposalStep,
    ResultStep,
    SummarizationStep,
    VFSStep,
    RelaxationStep,
    RequirementsStep
], Discriminator("type")]

audit = psycopg.connect(sys.argv[2])
db = AuditDB(audit)

thread_id = sys.argv[1]
(run_info, vfs_init) = db.get_run_info(thread_id=thread_id)

# Initialize VFS from audit DB
vfs = VFSManager({k: v.decode("utf-8") for (k, v) in vfs_init.to_dict().items()})


# ========================================
# Composer-Specific Tool Handlers
# ========================================

def handle_cvl_manual_search(step: dict, tool_id: str) -> SearchStep:
    """Handle cvl_manual_search tool case."""
    query = step["input"]["question"]
    assert isinstance(query, str)
    res_list: List[ManualResult] = []
    for res in db.get_manual_results(
        thread_id=sys.argv[1],
        tool_id=tool_id
    ):
        res_list.append(res)
    return SearchStep(
        query=query,
        results=res_list,
        type="search",
        vfs_snapshot=vfs.curr_version
    )

def handle_put_file(step: dict) -> PutFileStep:
    """Handle put_file tool case."""
    files = cast(Dict[str, str], step["input"]["files"])
    update: List[FileUpdate] = []
    for (k, v) in files.items():
        if k not in vfs.curr_data:
            update.append(FreshFile(
                path=k,
                contents=v,
                type="fresh"
            ))
        else:
            curr_version = vfs.curr_data[k]
            unified = compute_diff(k, curr_version=curr_version, new_version=v)
            update.append(Diff(
                type="diff",
                contents=v,
                diff_lines=unified,
                path=k
            ))
    result = PutFileStep(
        vfs_snapshot=vfs.curr_version,
        type="put_file",
        updates=update
    )
    vfs.push_update(files)
    return result

def handle_certora_prover(step: dict, tool_id: str, queue: MessageQueue) -> ProverStep:
    """Handle certora_prover tool case."""
    input_file = step["input"]["source_files"][0]
    rule = step["input"].get("rule", None)
    assert isinstance(input_file, str)

    nxt = queue.peek()
    todo_list : str | None = None
    if nxt is not None and isinstance(nxt, ToolMessage) and nxt.text() == "... Output truncated ...":
        queue.take()
        todo_msg = queue.take()
        assert isinstance(todo_msg, HumanMessage)
        assert type(todo_msg.content) is list
        todo_list_r = todo_msg.content[1]
        assert isinstance(todo_list_r, str)
        todo_list = todo_list_r

    results = list(db.get_rule_results(
        thread_id=sys.argv[1],
        tool_id=tool_id
    ))
    return ProverStep(
        type="prover",
        contract_file=input_file,
        results=results,
        rule=rule,
        vfs_snapshot=vfs.curr_version,
        todo_list=todo_list
    )

def handle_human_in_the_loop(step: dict, msg_queue: MessageQueue) -> QuestionStep:
    """Handle human_in_the_loop tool case."""
    question_input = step["input"]
    content = extract_human_response(msg_queue)
    assert isinstance(content, str)
    if content.startswith("Human Response: "):
        content = content[len("Human Response: "):]
    return QuestionStep(
        vfs_snapshot=vfs.curr_version,
        type="question",
        context=question_input["context"],
        query=question_input["question"],
        answer=content.strip(),
        code=question_input.get("code", None)
    )

def handle_propose_spec_change(step: dict, message_queue: MessageQueue) -> ProposalStep:
    """Handle propose_spec_change tool case."""
    question_input = step["input"]
    explanation = question_input["explanation"]
    proposed_spec = question_input["proposed_spec"]
    curr_version = vfs.curr_data["rules.spec"]
    diff = compute_diff("rules.spec", curr_version, proposed_spec)
    resp = extract_human_response(message_queue)
    result = ProposalStep(
        type="proposal",
        vfs_snapshot=vfs.curr_version,
        explanation=explanation,
        human_response=resp.strip(),
        proposed_diff=diff
    )
    if resp.startswith("ACCEPTED"):
        vfs.push_update({"rules.spec": proposed_spec})
    return result

def handle_code_result(step: dict) -> ResultStep:
    """Handle code_result tool case."""
    result_input = step["input"]
    return ResultStep(
        type="result",
        vfs_snapshot=vfs.curr_version,
        comments=result_input["comments"],
        files=result_input["source"]
    )

def handle_human_relaxation(step: dict, queue: MessageQueue) -> RelaxationStep:
    req_input = step["input"]
    response = extract_human_response(queue)
    return RelaxationStep(
        type="relaxation",
        vfs_snapshot=vfs.curr_version,
        response=response,
        requirement=req_input["req_text"],
        judgment=req_input["judgment"],
        context=req_input["context"],
        explanation=req_input["explanation"]
    )

def requirements_judge(step: dict, queue: MessageQueue) -> RequirementsStep:
    nxt = queue.take()
    assert isinstance(nxt, ToolMessage)
    t = nxt.text()
    
    # Parse stream of XML result elements
    import xml.etree.ElementTree as ET
    from html import unescape
    
    # Since the content is not XML escaped and may contain parse errors,
    # we need to handle it carefully. Split by </result> to get individual results
    results = []
    result_parts = t.split('</result>')
    
    for part in result_parts:
        part = part.strip()
        if not part:
            continue
            
        # Add back the closing tag
        if not part.endswith('</result>'):
            part += '</result>'
            
        # Find start of <result> tag
        start_idx = part.find('<result>')
        if start_idx == -1:
            continue
        part = part[start_idx:]
        req_text: str
        judgment: str
        try:
            # Try to parse as XML first
            root = ET.fromstring(part)
            requirement = root.find('requirement')
            classification = root.find('classification')
            comments = root.find('comments')
            
            req_text = requirement.text if requirement is not None and requirement.text is not None else ""
            judgment = classification.text if classification is not None and classification.text is not None else "IGNORED"
            commentary = comments.text if comments is not None else None
            
        except ET.ParseError:
            # Fallback: manual parsing for malformed XML
            req_text = ""
            judgment = "IGNORED"
            commentary = None
            
            # Extract requirement text
            req_start = part.find('<requirement>') + len('<requirement>')
            req_end = part.find('</requirement>')
            if req_start > len('<requirement>') - 1 and req_end != -1:
                req_text = part[req_start:req_end]
            
            # Extract classification
            class_start = part.find('<classification>') + len('<classification>')
            class_end = part.find('</classification>')
            if class_start > len('<classification>') - 1 and class_end != -1:
                judgment = part[class_start:class_end]
            
            # Extract comments if present
            comm_start = part.find('<comments>')
            if comm_start != -1:
                comm_start += len('<comments>')
                comm_end = part.find('</comments>')
                if comm_end != -1:
                    commentary = part[comm_start:comm_end]
        
        # Create ReqEval object
        req_eval = ReqEval(
            require_text=req_text,
            judgment=judgment if judgment in ["SATISFIED", "PARTIAL", "VIOLATED", "IGNORED", "LIKELY"] else "IGNORED" #type: ignore
        )
        if commentary:
            req_eval["commentary"] = commentary
            
        results.append(req_eval)
    
    return RequirementsStep(
        type="judge",
        vfs_snapshot=vfs.curr_version,
        evaluation=results
    )

def parse_message(checkpoint: CheckpointTuple) -> list[Steps]:
    state_messages = cast(list[BaseMessage], checkpoint.checkpoint["channel_values"]["messages"])
    queue = MessageQueue(state_messages)

    events : list[Steps] = []
    prev = checkpoint.parent_config
    while prev is not None:
        prev_tuple = checkpointer.get_tuple(prev)
        if prev_tuple is None:
            raise RuntimeError("odd")
        prev_id = prev_tuple.checkpoint["id"]
        summ = db.get_summary_after_checkpoint(
            thread_id=thread_id,
            checkpoint_id=prev_id
        )
        if summ is not None:
            prev_events = parse_message(prev_tuple)
            events.extend(prev_events)
            events.append(SummarizationStep(
                type="summarization",
                vfs_snapshot=vfs.curr_version,
                summary_md=summ
            ))
            break
        prev = prev_tuple.parent_config

    while queue.has_next():
        m = queue.take()
        if not isinstance(m, AIMessage):
            continue
        cont: List[str | dict]
        if isinstance(m.content, str):
            cont = [m.content]
        else:
            cont = m.content
        messages: List[AIStepMessage] = []
        for step in cont:
            if isinstance(step, str):
                messages.append(AIStepMessage(text=step, type="text")) # type: ignore
                continue
            ty = step.get("type")
            match ty:
                case "thinking":
                    messages.append({"type": "thinking", "text": step["thinking"]})
                case "text":
                    messages.append({"type": "text", "text": step["text"]})
                case "tool_use":
                    tool_id = step["id"]
                    which = step["name"]
                    events.append(AIStep(
                        vfs_snapshot=vfs.curr_version,
                        type="ai",
                        messages=messages,
                        tool=which
                    ))
                    match which:
                        case "cvl_manual_search":
                            events.append(handle_cvl_manual_search(step, tool_id))
                        case "put_file":
                            events.append(handle_put_file(step))
                        case "certora_prover":
                            events.append(handle_certora_prover(step, tool_id, queue))
                        case "human_in_the_loop":
                            events.append(handle_human_in_the_loop(step, queue))
                        case "propose_spec_change":
                            events.append(handle_propose_spec_change(step, queue))
                        case "code_result" | "result":
                            events.append(handle_code_result(step))
                        case "requirement_relaxation_request":
                            events.append(handle_human_relaxation(step, queue))
                        case "requirements_evaluation":
                            events.append(requirements_judge(step, queue))
                        case "get_file" | "list_files" | "grep_files":
                            events.append(VFSStep(
                                type="vfs",
                                vfs_snapshot=vfs.curr_version,
                                commands=handle_vfs_tools(step, queue)))
                        case _:
                            print("unhandled: " + which)
                            print(step)

    return events

checkpointer = get_checkpointer()

x = checkpointer.get_tuple({
    "configurable": {"thread_id": sys.argv[1]}
})

assert x is not None

spec_interface_swapped = run_info["spec"].basename.endswith(".sol")

events : list[Steps] = []
init = InitialStep(
    vfs_snapshot=0,
    type="initial",
    interface=run_info["interface"].string_contents if not spec_interface_swapped else run_info["spec"].string_contents,
    spec=run_info["spec"].string_contents if not spec_interface_swapped else run_info["interface"].string_contents ,
    system_doc=run_info["system"].string_contents
)

if (rq := run_info.get("reqs", None)) is not None:
    init["reqs"] = rq

events.append(
    init
)


events.extend(parse_message(x))
# print(json.dumps(evs, indent=2))

output = load_jinja_template("trace-explorer.html.j2", fs_dump=json.dumps(vfs.fs), steps_dump=json.dumps(events))

with open(sys.argv[3], "w") as out:
    out.write(output)

print("bye")
