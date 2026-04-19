"""
Shared CVL tools for spec generation workflows.

This module provides tools for writing CVL spec files,
shared between natspec (natural language spec generation) and
source_spec (source-based spec generation) workflows.
"""

import os
import subprocess
import tempfile
from importlib.resources import files
from typing import Annotated, Literal, overload
from typing_extensions import TypedDict

from langchain_core.tools import tool, InjectedToolCallId, BaseTool
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field, create_model

from composer.cvl.schema import CVLFile
from composer.cvl.pretty_print import pretty_print
from composer.ui.tool_display import tool_display_of, CommonTools, ToolDisplay, suppress_ack

from graphcore.graph import tool_state_update

_put_cvl_display = ToolDisplay(
    "Writing spec", suppress_ack("Spec write result")
)
_put_cvl_raw_display = _put_cvl_display

_get_cvl_display = ToolDisplay("Reading spec", None)


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
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".spec", delete=False) as f:
            f.write(pp)
            f.flush()
            if "CERTORA" in os.environ:
                certora_dir = os.environ["CERTORA"]
                emv_jar = os.path.join(certora_dir, "certora_jars", "Typechecker.jar")
            else:
                emv_jar = str(files("certora_jars") / "Typechecker.jar")
            res = subprocess.run(
                ["java", "-classpath", emv_jar, "EntryPointKt", f.name],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
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
    except Exception:
        return "Syntax checker failed"
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