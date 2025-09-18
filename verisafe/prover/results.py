from pydantic import Field, BaseModel, ValidationError
from typing import List, Optional, Dict
from pathlib import Path
import re
import json
from verisafe.prover.ptypes import RuleResult, StatusCodes

class RuleNodeModel(BaseModel):
    name: str = Field(description="The name of the node")
    output: List[str]
    children: List["RuleNodeModel"]
    status: Optional[str] = Field(description="The smt status")


class TreeViewStatus(BaseModel):
    rules: List[RuleNodeModel]


class SarifArgs(BaseModel):
    value: str
    # ignoring the other fields


class MessageModel(BaseModel):
    text: str
    arguments: List[SarifArgs]


class CallTraceModel(BaseModel):
    message: MessageModel
    childrenList: List["CallTraceModel"]

def read_and_format_run_result(s: Path) -> Dict[str, RuleResult] | str:
    tree_view_dir = s / "Reports" / "treeView"
    status_files = tree_view_dir.glob("treeViewStatus_*.json")

    search_patt = re.compile(
        r'treeViewStatus_(\d+).json'
    )

    max_n = -1
    for p in status_files:
        if p.name is None:
            continue
        match = search_patt.match(p.name)
        if match is None:
            continue
        index = int(match.group(1))
        if index < max_n:
            continue
        max_n = index
    if max_n == -1:
        return "Certora prover returned no results: this is likely a bug"

    final_status = s / "Reports" / "treeView" / f"treeViewStatus_{max_n}.json"
    with open(final_status, "r") as result_file:
        run_status = json.load(result_file)
    try:
        loaded_data = TreeViewStatus.model_validate(run_status)
    except ValidationError:
        return "Certora prover returned malformed tree view data: this is likely a bug"

    to_ret: Dict[str, RuleResult] = {}
    for r in loaded_data.rules:
        to_ret[r.name] = dump_tree_view_node(tree_view_dir, r)
    return to_ret


def dump_tree_view_node(context: Path, r: RuleNodeModel) -> RuleResult:
    status_string: StatusCodes
    if r.status is not None:
        match r.status:
            case "VIOLATED" | "VERIFIED" | "TIMEOUT":
                status_string = r.status
            case _:
                status_string = "ERROR"
    else:
        status_string = "ERROR"

    cex_dump: Optional[str] = None
    if status_string == "VIOLATED" and len(r.output) > 0:
        assert len(r.output) == 1
        with open(context / r.output[0], "r") as cex:
            dump_model = json.load(cex)
        assert isinstance(dump_model, dict)
        if "callTrace" in dump_model:
            cex_node = CallTraceModel.model_validate(dump_model["callTrace"])
            cex_dump = "<counterexample>" + calltrace_to_xml(cex_node) + "</counterexample>"
    return RuleResult(
        status=status_string,
        cex_dump=cex_dump,
        name=r.name
    )

def calltrace_to_xml(node: CallTraceModel) -> str:
    """
    Convert a tree-like JSON node to XML format.

    Args:
        node: A dictionary with 'message' field and optional 'childrenList' field

    Returns:
        String representation of the XML
    """
    # Extract and format the message

    # Replace placeholders with argument values
    formatted_message = node.message.text
    for i, arg in enumerate(node.message.arguments):
        placeholder = f"{{{i}}}"
        formatted_message = formatted_message.replace(placeholder, arg.value)

    # Start building XML
    xml_parts = [f"<message>{formatted_message}</message>"]

    # Process children if they exist
    for child in node.childrenList:
        # skip this, avoid confusing the llm
        if child.message.text == "Setup" or child.message.text == "Global State" or child.message.text == "Evaluate branch condition":
            continue
        child_xml = calltrace_to_xml(child)
        xml_parts.append(f"<child>{child_xml}</child>")

    return "".join(xml_parts)
