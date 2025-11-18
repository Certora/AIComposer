from typing import Optional, Callable, TypeVar
from typing_extensions import Iterable
from pathlib import Path
import re
import json

from pydantic import Field, BaseModel, ValidationError

from verisafe.prover.ptypes import RuleResult, StatusCodes, RulePath

T = TypeVar('T')
R = TypeVar('R')

class RuleNodeModel(BaseModel):
    name: str = Field(description="The name of the node")
    output: list[str]
    children: list["RuleNodeModel"]
    status: Optional[str] = Field(description="The smt status")
    nodeType: str


class TreeViewStatus(BaseModel):
    rules: list[RuleNodeModel]


class SarifArgs(BaseModel):
    value: str
    # ignoring the other fields


class MessageModel(BaseModel):
    text: str
    arguments: list[SarifArgs]


class CallTraceModel(BaseModel):
    message: MessageModel
    childrenList: list["CallTraceModel"]


def _flat_yield(curr: Iterable[T], gen: Callable[[T], Iterable[R]]) -> Iterable[R]:
    for t in curr:
        for to_yield in gen(t):
            yield to_yield

def _to_status_string(s: str | None) -> StatusCodes:
    if s is None:
        return "ERROR"
    match s:
        case "VIOLATED" | "VERIFIED" | "TIMEOUT" | "SANITY_FAILED":
            return s
        case _:
            return "ERROR"


def flatten_tree_view_root(context: Path, r: RuleNodeModel) -> Iterable[RuleResult]:
    assert r.nodeType == "ROOT"
    return flatten_tree_view(context, r, RulePath(rule=r.name))

def flatten_tree_view(context: Path, r: RuleNodeModel, path: RulePath) -> Iterable[RuleResult]:
    stat = _to_status_string(r.status)
    effective_path = path
    if r.nodeType == "METHOD_INSTANTIATION":
        effective_path = effective_path.copy(method=r.name)
    elif r.nodeType == "CONTRACT":
        effective_path = effective_path.copy(contract = r.name)
    elif r.nodeType == "INVARIANT_SUBCHECK":
        if "constructor" in r.name:
            effective_path = effective_path.copy(method="constructor")

    if stat == "ERROR":
        return [RuleResult(
            path=effective_path,
            cex_dump=None,
            status=stat
        )]
    if stat == "VERIFIED":
        non_sanity_children = any([ c.nodeType != "SANITY" for c in r.children ])
        if non_sanity_children:
            return _flat_yield(r.children, lambda c: flatten_tree_view(context, c, effective_path))
        else:
            return [RuleResult(
                path=effective_path,
                cex_dump=None,
                status=stat
            )]
    
    if stat == "TIMEOUT":
        if len(r.children) == 0:
            return [RuleResult(path=effective_path, cex_dump=None,status=stat)]
    assert stat == "TIMEOUT" or stat == "VIOLATED" or stat == "SANITY_FAILED"
    violated_assert_children = any([ c.nodeType == "VIOLATED_ASSERT" for c in r.children])
    if violated_assert_children:
        assert stat == "VIOLATED" and len(r.output) > 0
        output_file = r.output[0]
        dump_model = json.loads((context / output_file).read_text())
        cex_dump : None | str = None
        assert isinstance(dump_model, dict)
        if "callTrace" in dump_model:
            cex_node = CallTraceModel.model_validate(dump_model["callTrace"])
            cex_dump = "<counterexample>" + calltrace_to_xml(cex_node) + "</counterexample>"
        return [RuleResult(
            path = effective_path,
            cex_dump=cex_dump,
            status=stat
        )]
    if r.nodeType == "SANITY":
        assert stat == "SANITY_FAILED"
        return [RuleResult(
            path=effective_path,
            cex_dump=None,
            status=stat
        )]
    return _flat_yield(r.children, lambda c: flatten_tree_view(context, c, effective_path))

class NoTreeViewResultError(RuntimeError):
    def __init__(self, where: Path):
        super().__init__(f"No tree views found in {where}")

class MalformedTreeVew(RuntimeError):
    def __init__(self, wrapped: ValidationError):
        super().__init__(wrapped)


def get_final_treeview(s: Path) -> tuple[TreeViewStatus, Path]:
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
        raise NoTreeViewResultError(s)

    final_status = s / "Reports" / "treeView" / f"treeViewStatus_{max_n}.json"
    with open(final_status, "r") as result_file:
        run_status = json.load(result_file)
    try:
        loaded_data = TreeViewStatus.model_validate(run_status)
        return (loaded_data, tree_view_dir)
    except ValidationError as e:
        raise MalformedTreeVew(e)


def read_and_format_run_result(s: Path) -> dict[str, RuleResult] | str:
    loaded_data : TreeViewStatus
    tree_view_dir: Path
    try:
        (loaded_data, tree_view_dir) = get_final_treeview(s)
    except NoTreeViewResultError:
        return "Certora prover returned no results: this is likely a bug"
    except MalformedTreeVew:
        return "Certora prover returned malformed tree view data: this is likely a bug"

    to_ret: dict[str, RuleResult] = {}
    for r in _flat_yield(loaded_data.rules, lambda r: flatten_tree_view_root(tree_view_dir, r)):
        to_ret[r.name] = r
    return to_ret

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
