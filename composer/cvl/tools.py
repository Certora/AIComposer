"""
Shared CVL tools for spec generation workflows.

This module provides tools for writing CVL spec files,
shared between natspec (natural language spec generation) and
source_spec (source-based spec generation) workflows.
"""

import logging
import subprocess
import tempfile
from typing import Annotated, Literal, overload
from typing_extensions import TypedDict

from langchain_core.tools import tool, InjectedToolCallId, BaseTool
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field, create_model

from composer.certora_env import typechecker_jar
from composer.core.edit import replace_unique, EditOk, EditErr
from composer.cvl.schema import CVLFile
from composer.cvl.pretty_print import pretty_print
from composer.ui.tool_display import tool_display_of, CommonTools, ToolDisplay, suppress_ack

from graphcore.graph import tool_state_update

_logger = logging.getLogger(__name__)

_put_cvl_display = ToolDisplay(
    "Writing spec", suppress_ack("Spec write result")
)
_put_cvl_raw_display = _put_cvl_display

_get_cvl_display = ToolDisplay("Reading spec", None)

_edit_cvl_display = ToolDisplay("Editing spec", suppress_ack("Spec edit result"))


put_cvl_description = """
Put a new version of the proposed spec file onto the VFS. The tool schema constrains
you to putting only syntactically valid CVL. However, a pretty printed version of this syntax
is ultimately what is saved on the VFS.

This pretty printed file is then run through the official CVL parser. If the code fails to parse,
this tool will reject the update, with the reported errors.
"""


class PutCVLSchemaModel(BaseModel):
    cvl_file: CVLFile = Field(description="The CVL AST to put in the VFS")


class PutCVLSchemaLG(BaseModel):
    cvl_file: dict = Field(description="The CVL AST to put in the VFS")
    tool_call_id: Annotated[str, InjectedToolCallId]


PutCVLSchemaLG.__doc__ = put_cvl_description

DEFAULT_READ_KEY = "did_read"

DEFAULT_SPEC_KEY = "curr_spec"


class PutCVLRaw(BaseModel):
    """
    A version of put CVL which accepts the surface syntax of CVL. You should only use
    this if you have extremely high confidence that the CVL representation you are passing in
    is correct.

    If `cvl_file` is determined to have a syntax error, this update is rejected.
    """
    cvl_file: str = Field(description="The raw, surface syntax of the CVL file.")
    tool_call_id: Annotated[str, InjectedToolCallId]


def maybe_update_cvl(
    *,
    tool_call_id: str,
    pp: str,
    spec_key: str,
    ast_json: dict | None = None,
    reset_read: str | None = None
) -> str | Command:
    """
    Validate CVL syntax and update state if valid.

    Uses the Certora emv.jar parser to validate the CVL syntax.
    Returns a Command to update state on success, or an error message on failure.
    """
    # Resolve the typechecker jar and run it. A failure in either step is an
    # environment/plumbing problem (jar not packaged, CERTORA misconfigured, java
    # not on PATH), NOT a spec error — surface it distinctly so the caller stops
    # trying to "fix" valid CVL, and log the real exception for the operator.
    try:
        emv_jar = str(typechecker_jar())
        with tempfile.NamedTemporaryFile("w", suffix=".spec", delete=False) as f:
            f.write(pp)
            f.flush()
            res = subprocess.run(
                ["java", "-classpath", emv_jar, "EntryPointKt", f.name],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
    except Exception as exc:
        _logger.exception("CVL syntax checker could not be launched")
        return (
            "Syntax checker could not be launched: "
            f"{type(exc).__name__}: {exc}. This is an environment problem, not a "
            "problem with the spec — do not keep retrying; surface it to the operator."
        )

    if res.returncode != 0:
        import json as _json
        with tempfile.NamedTemporaryFile("w", suffix=".spec", prefix="pp_fail_", delete=False, dir="/tmp") as dbg_pp:
            dbg_pp.write(pp)
        if ast_json is not None:
            with tempfile.NamedTemporaryFile("w", suffix=".json", prefix="pp_fail_", delete=False, dir="/tmp") as dbg_json:
                _json.dump(ast_json, dbg_json, indent=2)
        return f"""
Update rejected, the syntax checker exited with non-zero status

stdout:
{res.stdout}

stderr:
{res.stderr}
"""
    update = {}
    update[spec_key] = pp
    if reset_read:
        update[reset_read] = False
    return tool_state_update(
        tool_call_id=tool_call_id,
        content="Accepted",
        **update
    )


@tool_display_of(_put_cvl_display)
@tool(args_schema=PutCVLSchemaLG)
def put_cvl(
    cvl_file: dict,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command | str:
    """Put a CVL file using the structured AST representation."""
    pp: str
    try:
        pp = pretty_print(CVLFile.model_validate(cvl_file))
    except Exception:
        return "Failed to pretty print the AST"
    return maybe_update_cvl(tool_call_id=tool_call_id, pp=pp, ast_json=cvl_file, reset_read=DEFAULT_READ_KEY, spec_key=DEFAULT_SPEC_KEY)

@tool_display_of(_put_cvl_raw_display)
@tool(args_schema=PutCVLRaw)
def put_cvl_raw(
    tool_call_id: Annotated[str, InjectedToolCallId],
    cvl_file: str
) -> str | Command:
    """Put a CVL file using raw surface syntax."""
    return maybe_update_cvl(tool_call_id=tool_call_id, pp=cvl_file, reset_read=DEFAULT_READ_KEY, spec_key=DEFAULT_SPEC_KEY)

class WithCurrSpec(TypedDict):
    curr_spec: str | None

class WithCurrSpecAndDidRead(WithCurrSpec):
    did_read: bool

class WithCurrSpecNonNull(TypedDict):
    curr_spec: str

class GetCVLSchemaTemplate(BaseModel):
    """
    Retrive the textual representation of the current specification.
    """

@overload
def get_cvl[S: WithCurrSpecAndDidRead](
    ty: type[S],
    *,
    set_did_read: Literal[True],
) -> BaseTool: ...

@overload
def get_cvl[S: WithCurrSpecNonNull](
    ty: type[S],
) -> BaseTool: ...


@overload
def get_cvl[S: WithCurrSpec](
    ty: type[S],
) -> BaseTool: ...

def get_cvl(
    ty: type,
    *,
    set_did_read: bool = False,
) -> BaseTool:
    extra_fields: dict = {}
    if set_did_read:
        extra_fields["tool_call_id"] = (Annotated[str, InjectedToolCallId], ...)
    schema = create_model(
        "GetCVL",
        __base__=GetCVLSchemaTemplate,
        __doc__=GetCVLSchemaTemplate.__doc__,
        state=(Annotated[ty, InjectedState], ...),
        **extra_fields,
    )
    @tool_display_of(_get_cvl_display)
    @tool(args_schema=schema)
    def get_cvl(
        **args
    ) -> str | Command:
        st = args["state"]
        if st["curr_spec"] is None:
            return "No spec file written yet"
        spec = st["curr_spec"]
        if set_did_read:
            update = {
                DEFAULT_READ_KEY: True
            }
            return tool_state_update(
                tool_call_id=args["tool_call_id"],
                content=spec,
                **update
            )
        return spec
    return get_cvl


edit_cvl_description = """
Make a surgical edit to the current CVL spec instead of re-emitting the whole file.

Provide `old_string` — an exact span copied from the current spec — and `new_string`
to replace it with. `old_string` must occur exactly once; include enough surrounding
context to make it unique. Prefer this over `put_cvl_raw` when changing only part of an
existing spec (e.g. one line of a failing rule): it is dramatically cheaper than
re-sending the entire file.

The edited spec is run through the CVL parser exactly like `put_cvl_raw`. If the result
fails to parse, the edit is rejected with the parser errors and the buffer is unchanged.
"""


class EditCVLTemplate(BaseModel):
    old_string: str = Field(
        description="The exact span of the current spec to replace. Must occur exactly once; "
        "include surrounding context to disambiguate."
    )
    new_string: str = Field(description="The text to replace `old_string` with.")


def edit_cvl[S: WithCurrSpec](ty: type[S]) -> BaseTool:
    """A surgical-edit tool over the ``curr_spec`` buffer: single-occurrence
    string replace, then re-validate the result exactly like ``put_cvl_raw``."""
    schema = create_model(
        "EditCVL",
        __base__=EditCVLTemplate,
        __doc__=edit_cvl_description,
        state=(Annotated[ty, InjectedState], ...),
        tool_call_id=(Annotated[str, InjectedToolCallId], ...),
    )

    @tool_display_of(_edit_cvl_display)
    @tool(args_schema=schema)
    def edit_cvl(**args) -> str | Command:
        st = args["state"]
        if st["curr_spec"] is None:
            return "No spec file written yet — use put_cvl or put_cvl_raw first."
        match replace_unique(st["curr_spec"], args["old_string"], args["new_string"]):
            case EditErr(message=msg):
                return msg
            case EditOk(text=new_text):
                return maybe_update_cvl(
                    tool_call_id=args["tool_call_id"],
                    pp=new_text,
                    spec_key=DEFAULT_SPEC_KEY,
                    reset_read=DEFAULT_READ_KEY,
                )
    return edit_cvl