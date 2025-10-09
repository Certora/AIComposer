import pathlib
import sys

if __name__ != "__main__":
    raise RuntimeError("This is a script only module")

import sqlite3
import difflib
import json

from typing import Dict, Optional, List, cast, TypedDict, Literal, Annotated, Union, TypeVar, Generic
from langgraph.checkpoint.base import CheckpointTuple
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from pydantic import Discriminator

verisafe_dir = str(pathlib.Path(__file__).parent.parent.parent.absolute())

if verisafe_dir not in sys.path:
    sys.path.append(verisafe_dir)

from verisafe.audit.types import ManualResult, RuleResult
from verisafe.audit.db import AuditDB
from verisafe.workflow.factories import get_checkpointer
from verisafe.templates.loader import load_jinja_template


checkpointer = get_checkpointer()

x = checkpointer.get_tuple({
    "configurable": {"thread_id": sys.argv[1]}
})

def get_initial_state(check: CheckpointTuple) -> Optional[Dict[str, str]]:
    if check.parent_config is not None:
        parent_state = checkpointer.get_tuple(check.parent_config)
        if parent_state is not None:
            parent_res = get_initial_state(parent_state)
            if parent_res is not None:
                return parent_res
    
    return check.checkpoint["channel_values"].get("virtual_fs", None)

assert x is not None
initial_fs = get_initial_state(x)

class VFSManager():
    def __init__(self, ver_0: Dict[str, str]):
        self.fs = [ver_0]
        self.curr_data = ver_0.copy()

    def push_update(self, upd: Dict[str, str]):
        self.fs.append(upd.copy())
        for (k,v) in upd.items():
            self.curr_data[k] = v

    @property
    def curr_version(self) -> int:
        return len(self.fs) - 1

assert initial_fs is not None
vfs = VFSManager(initial_fs)

msgs = x.checkpoint["channel_values"]["messages"]
msg_list = cast(List[BaseMessage], msgs)

StepTy = TypeVar("StepTy", 
                 Literal["initial"],
                 Literal["ai"],
                 Literal["prover"],
                 Literal["search"],
                 Literal["put_file"],
                 Literal["question"],
                 Literal["proposal"],
                 Literal["result"])

class AbstractStep(TypedDict, Generic[StepTy]):
    vfs_snapshot: int
    type: StepTy


class InitialStep(AbstractStep[Literal["initial"]]):
    spec: str
    interface: str
    system_doc: str

class AIStepMessage(TypedDict):
    type: Literal["thinking", "text"]
    text: str

class AIStep(AbstractStep[Literal["ai"]]):
    messages: List[AIStepMessage]
    tool: str

class ProverStep(AbstractStep[Literal["prover"]]):
    contract_file: str
    rule: Optional[str]
    results: List[RuleResult]

class SearchStep(AbstractStep[Literal["search"]]):
    query: str
    results: List[ManualResult]

class QuestionStep(AbstractStep[Literal["question"]]):
    query: str
    answer: str
    context: str
    code: Optional[str]

class ResultStep(AbstractStep[Literal["result"]]):
    comments: str
    files: List[str]

class ProposalStep(AbstractStep[Literal["proposal"]]):
    explanation: str
    human_response: str
    proposed_diff: List[str]
    
class FreshFile(TypedDict):
    type: Literal["fresh"]
    contents: str
    path: str

class Diff(TypedDict):
    type: Literal["diff"]
    path: str
    contents: str
    diff_lines: List[str]

FileUpdate = Annotated[Union[Diff, FreshFile], Discriminator("type")]

class PutFileStep(AbstractStep[Literal["put_file"]]):
    updates: List[FileUpdate]

Steps = Annotated[Union[
    AIStep,
    InitialStep,
    ProverStep,
    SearchStep,
    PutFileStep,
    QuestionStep,
    ProposalStep,
    ResultStep
], Discriminator("type")]

events: List[Steps] = []

audit = sqlite3.connect(sys.argv[2])
db = AuditDB(audit)

thread_id = sys.argv[1]
run_info = db.get_run_info(thread_id=thread_id)

events.append(InitialStep(
    vfs_snapshot=0,
    type="initial",
    interface=run_info["interface"]["content"],
    spec=run_info["spec"]["content"],
    system_doc=run_info["system"]["content"]
))

i = 0
def compute_diff(path: str, curr_version: str, new_version: str) -> List[str]:
    ud = difflib.unified_diff(
        a=curr_version.splitlines(keepends=True),
        b=new_version.splitlines(keepends=True),
        fromfile="a/" + path,
        tofile="b/" + path
    )
    return list(ud)

def extract_human_response(msgs: List[BaseMessage], nxt_index: int) -> str:
    answer_msg = msgs[nxt_index]
    assert isinstance(answer_msg, ToolMessage)
    content: str
    match answer_msg.content:
        case list():
            first_elem = answer_msg.content[0]
            if isinstance(first_elem, dict):
                content = first_elem["text"]
            else:
                content = first_elem
        case str():
            content = answer_msg.content
        case _:
            raise RuntimeError(f"Unexpected type {type(answer_msg.content)}")
    return content

while i < len(msgs):
    m = msgs[i]
    match m:
        case AIMessage():
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
                                query = step["input"]["question"]
                                assert isinstance(query, str)
                                res_list: List[ManualResult] = []
                                for res in db.get_manual_results(
                                    thread_id=sys.argv[1],
                                    tool_id=tool_id
                                ):
                                    res_list.append(res)
                                events.append(SearchStep(
                                    query=query,
                                    results=res_list,
                                    type="search",
                                    vfs_snapshot=vfs.curr_version
                                ))
                            case "put_file":
                   
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
                                events.append(PutFileStep(
                                    vfs_snapshot=vfs.curr_version,
                                    type="put_file",
                                    updates=update
                                ))
                                vfs.push_update(files)
                            case "certora_prover":
                                input_file = step["input"]["source_files"][0]
                                rule = step["input"].get("rule", None)
                                assert isinstance(input_file, str)
                                results = list(db.get_rule_results(
                                    thread_id=sys.argv[1],
                                    tool_id=tool_id
                                ))
                                events.append(ProverStep(
                                    type="prover",
                                    contract_file=input_file,
                                    results=results,
                                    rule=rule,
                                    vfs_snapshot=vfs.curr_version
                                ))
                            case "human_in_the_loop":
                                question_input = step["input"]
                                nxt_index = i+1
                                content = extract_human_response(msgs, nxt_index)
                                assert isinstance(content, str)
                                if content.startswith("Human Response: "):
                                    content = content[len("Human Response: "):]
                                events.append(QuestionStep(
                                    vfs_snapshot=vfs.curr_version,
                                    type="question",
                                    context=question_input["context"],
                                    query=question_input["question"],
                                    answer=content.strip(),
                                    code=question_input.get("code", None)
                                ))
                            case "propose_spec_change":
                                question_input = step["input"]
                                explanation = question_input["explanation"]
                                proposed_spec = question_input["proposed_spec"]
                                curr_version = vfs.curr_data["rules.spec"]
                                diff = compute_diff("rules.spec", curr_version, proposed_spec)
                                resp = extract_human_response(
                                        msgs=msgs,
                                        nxt_index=i+1
                                )
                                events.append(ProposalStep(
                                    type="proposal",
                                    vfs_snapshot=vfs.curr_version,
                                    explanation=explanation,
                                    human_response=resp.strip(),
                                    proposed_diff=diff
                                ))
                                if resp.startswith("ACCEPTED"):
                                    vfs.push_update({"rules.spec": proposed_spec})
                            case "code_result":
                                result_input = step["input"]
                                events.append(ResultStep(
                                    type="result",
                                    vfs_snapshot=vfs.curr_version,
                                    comments=result_input["comments"],
                                    files=result_input["source"]
                                ))
                            case _:
                                print("unhandled: " + which)
                                print(step)
                        i+=1
                        break
        case _:
            pass
    i+=1

output = load_jinja_template("trace-explorer.html.j2", fs_dump=json.dumps(vfs.fs), steps_dump=json.dumps(events))

with open(sys.argv[3], "w") as out:
    out.write(output)

print("bye")