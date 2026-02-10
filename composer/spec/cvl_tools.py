"""
Shared CVL tools for spec generation workflows.

This module provides tools for writing CVL spec files,
shared between natspec (natural language spec generation) and
source_spec (source-based spec generation) workflows.
"""

import os
import subprocess
import tempfile
from typing import Annotated, TypedDict, NotRequired, cast

from langchain_core.tools import tool, InjectedToolCallId, BaseTool
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field, create_model

from composer.cvl.schema import CVLFile
from composer.cvl.pretty_print import pretty_print
from graphcore.graph import tool_state_update


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


class PutCVLRaw(BaseModel):
    """
    A version of put CVL which accepts the surface syntax of CVL. You should only use
    this if you have extremely high confidence that the CVL representation you are passing in
    is correct.

    If `cvl_file` is determined to have a syntax error, this update is rejected.
    """
    cvl_file: str = Field(description="The raw, surface syntax of the CVL file.")
    tool_call_id: Annotated[str, InjectedToolCallId]


def _maybe_update_cvl(
    tool_call_id: str,
    pp: str
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
            certora_dir = os.environ["CERTORA"]
            emv_jar = os.path.join(certora_dir, "emv.jar")
            res = subprocess.run(
                ["java", "-classpath", emv_jar, "spec.ParseCheckerKt", f.name],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            if res.returncode != 0:
                return f"""
Update rejected, the syntax checker exited with non-zero status

stdout:
{res.stdout}

stderr:
{res.stderr}
"""
    except Exception:
        return "Syntax checker failed"
    return tool_state_update(
        tool_call_id=tool_call_id,
        content="Accepted",
        curr_spec=pp
    )


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
    return _maybe_update_cvl(tool_call_id, pp)


@tool(args_schema=PutCVLRaw)
def put_cvl_raw(
    tool_call_id: Annotated[str, InjectedToolCallId],
    cvl_file: str
) -> str | Command:
    """Put a CVL file using raw surface syntax."""
    return _maybe_update_cvl(tool_call_id, cvl_file)

class WithCurrSpec(TypedDict):
    curr_spec: NotRequired[str]

class GetCVLSchemaTemplate(BaseModel):
    """
    Retrive the textual representation of the current specification.
    """

def get_cvl[S: WithCurrSpec](
    ty: type[S]
) -> BaseTool:
    schema = create_model(
        "GetCVL",
        __base__=GetCVLSchemaTemplate,
        __doc__=GetCVLSchemaTemplate.__doc__,
        state=Annotated[ty, InjectedState]
    )
    @tool(args_schema=schema)
    def get_cvl(
        **args
    ) -> str:
        st = cast(S, args["state"])
        if "curr_spec" not in st:
            return "No spec file written yet"
        else:
            return st["curr_spec"]
    return get_cvl