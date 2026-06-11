from dataclasses import dataclass
from typing import Callable, Literal, override, Annotated, Literal
from pydantic import Field, BaseModel, Discriminator

from typing_extensions import TypedDict

from langchain_core.tools import BaseTool

from langgraph.types import Command

from graphcore.summary import SummaryConfig
from graphcore.graph import tool_state_update
from graphcore.tools.schemas import WithImplementation, WithInjectedId, WithInjectedState, WithAsyncDependencies
from composer.spec.cvl_generation import (
    static_tools, run_cvl_generator, CVLGenerationInput, CVLGenerationState,
    FeedbackToolContext, check_completion, CVLGenerationExtra
)


from composer.spec.context import (
    WorkflowContext, CVLGeneration, SystemDoc
)
from composer.spec.prop import PropertyFormulation
from composer.spec.feedback import property_feedback_judge, Properties, FeedbackTemplate
from composer.spec.gen_types import TypedTemplate
from composer.spec.system_model import ContractComponentInstance, ContractName
from composer.spec.cvl_generation import CVL_JUDGE_KEY, FeedbackToolContext, static_tools, SkippedProperty
from composer.spec.service_host import ServiceHost
from composer.ui.tool_display import tool_display, suppress_ack
from composer.spec.natspec.task_description import Assembler, ConfigurationBuilder
from composer.spec.natspec.typecheck import TypeChecker

class SourceGenerationParams(Properties):
    context: ContractComponentInstance
    sort: Literal["greenfield", "existing", "update"]

NoSourceGen = TypedTemplate[SourceGenerationParams]("nosource_property_generation_prompt.j2")

class _CVLConfig(SummaryConfig[CVLGenerationState]):
    def __init__(self, contract_name: ContractName, stub_path: str):
        super().__init__(enabled=True)
        self.contract_name = contract_name
        self.stub_path = stub_path

    @override
    def get_summarization_prompt(self, state: CVLGenerationState) -> str:
        return """
You are approaching the context limit for your task. After this point, your context will be cleared
and the task restarted from the initial prompt.

To enable you to continue to work effectively after this compaction, summarize the current state of your task. In particular, summarize:
1. Any key findings about CVL you received from the CVL researcher or your own research
2. The current state of your task, including:
   a. What properties have been formalized
   b. What properties you have skipped, and why
   c. What properties have been accepted by the feedback tool.
3. If you have any outstanding, unaddressed feedback from your last iteration with the feedback tool, include that unaddressed feedback in your summary
4. Any techniques/attempts that you or the feedback rejected or didn't work

In other words, your summary should include all information necessary to prevent the next iteration on this task from repeating work
or repeating mistakes.

If your current task itself began with a summary, include the salient parts of that summary in your new summary.
"""

    @override
    def get_resume_prompt(self, state: CVLGenerationState, summary: str) -> str:
        return f"""
You are resuming this task already in progress. The current version of your spec (if any) is available via the `get_cvl` tool.

The current content of the type checking stub for the {self.contract_name} contract can be found at {self.stub_path}

A summary of your work up until this point is as follows:

BEGIN SUMMARY:
{summary}

END SUMMARY

**IMPORTANT**: Absolutely *nothing* has changed since the summary was produced and now. You do *NOT* need to reverify
any information about CVL present in your summary unless you discovery something *new* with necessitates revisiting those conclusions.
If you have outstanding feedback to address, you do *NOT* need to re-invoke the feedback tool; proceed immediately with addressing
that feedback.
"""

class NatspecGenerationExtra(TypedDict):
    failed: bool | None
    suggested_spec_path: str | None

class NatspecGenerationState(CVLGenerationState, NatspecGenerationExtra):
    pass

class NatspecGenerationInput(CVLGenerationInput, NatspecGenerationExtra):
    pass

class GenerationSuccess(BaseModel):
    commentary: str
    skipped: list[SkippedProperty]
    spec: str
    suggested_path: str
    ty: Literal["success"]

class GaveUp(BaseModel):
    reason: str
    ty: Literal["fail"]

class AuthorResult(BaseModel):
    result_wrapped: Annotated[GaveUp | GenerationSuccess, Discriminator("ty")]

@tool_display(
    label=lambda p: f"Giving up on property generation: {p['reason']}",
    result=None,
)
class GiveUpTool(WithImplementation[Command], WithInjectedId):
    """
    Call this tool to give up on the property generation for this task.

    This should only ever be called as a LAST RESORT when you have exhausted all other
    mechanisms to complete your task
    """
    reason : str = Field(description="The reason for giving up on your task")

    @override
    def run(self) -> Command:
        return tool_state_update(
            self.tool_call_id,
            "Accepted",
            failed=True,
            result=self.reason
        )
    
@dataclass
class ContractConfiguration:
    assembler: Assembler
    conf: ConfigurationBuilder


@tool_display(
    label="Delivering Spec",
    result=suppress_ack(label="Result call rejected")
)
class PublishTool(WithAsyncDependencies[Command | str, TypeChecker], WithInjectedId, WithInjectedState[CVLGenerationExtra]):
    """
    Call this tool when the feedback tool has accepted your specification and it is known to be type correct.
    
    This update will be rejected if the spec is *not* type correct, or the feedback tool has not
    given a "good" judgment on your spec.
    """
    commentary: str = Field(description="Human readable commentary on your specification.")
    suggested_spec_name : str = Field(
        description="The suggested name to use for your generated spec file. " \
        "Do *NOT* include the `.spec` extension. Prefer underscores over spaces, no special characters, etc.")

    async def run(self) -> Command | str:
        if self.state["curr_spec"] is None:
            return "No spec put yet"
        if (msg := check_completion(self.state)) is not None:
            return msg
        with self.tool_deps() as checker:
            tycheck = await checker(
                self.state["curr_spec"]
            )
            if tycheck is not None:
                return f"Completion rejected; spec typechecking failed: {tycheck}"
        return tool_state_update(
            self.tool_call_id, "Accepted",
            result=self.commentary,
            failed=False,
            suggested_spec_path=f"{self.suggested_spec_name}.spec"
        )

@tool_display("Type-checking spec", "Type-check result")
class AdvisoryTypecheck(WithAsyncDependencies[str, TypeChecker], WithInjectedState[CVLGenerationExtra]):
    """Run the CVL typechecker on your current working specification
    against the assembled project. This is advisory — use it to catch
    issues before attempting to finalize. Reads the current spec from
    state (written via ``put_cvl`` / ``put_cvl_raw``).
    """

    @override
    async def run(self) -> str:
        spec = self.state.get("curr_spec")
        if spec is None:
            return "No spec written yet. Use put_cvl or put_cvl_raw first."
        with self.tool_deps() as typecheck_spec:    
            result = await typecheck_spec(
                spec
            )
            if result is None:
                return "Typecheck passed."
            return f"Typecheck failed:\n{result}"


async def generate_cvl_batch(
    root_ctx: WorkflowContext[AuthorResult],
    env: ServiceHost,
    system_doc: SystemDoc,

    props: list[PropertyFormulation],
    component: ContractComponentInstance,
    contract_name: ContractName,

    typechecker: TypeChecker,

    injected_tools: list[BaseTool],
    stub_reader: Callable[[], str],
    stub_path: str,
) -> GenerationSuccess | GaveUp:
    if (cached := await root_ctx.cache_get(AuthorResult)) is not None:
        return cached.result_wrapped

    def stub_feedback_extras() -> list[str | dict]:
        return [
            f"The current typechecking stub for the {contract_name} contract is",
            stub_reader(),
            "For reference, the system document for the application is",
            system_doc.content.to_dict(),
        ]

    ctx = root_ctx.abstract(CVLGeneration)

    feedback_ctxt = property_feedback_judge(
        ctx=ctx.child(CVL_JUDGE_KEY), env=env, prompt=FeedbackTemplate.bind({
            "context": component,
            "sort": env.sort,
        }).depends(Properties), props=props, extra_inputs=stub_feedback_extras
    )

    g = (
        env.builder
        .with_tools(env.all_tools)
        .with_tools(injected_tools)
        .with_tools(static_tools())
        .with_tools([
            GiveUpTool.as_tool("give_up"),
            AdvisoryTypecheck.bind(typechecker).as_tool("advisory_typecheck"),
            PublishTool.bind(typechecker).as_tool("publish"),
            ctx.get_memory_tool()
        ])
        .with_output_key("result")
        .with_input(NatspecGenerationInput)
        .with_state(NatspecGenerationState)
        .with_context(FeedbackToolContext)
        .with_sys_prompt_template("nosource_property_generation_system_prompt.j2")
        .inject(
            lambda b: NoSourceGen.bind({
                "context": component,
                "properties": props,
                "sort": env.sort,
            }).render_to(b.with_initial_prompt_template)
        ).with_summary_config(_CVLConfig(contract_name, stub_path)).compile_async()
    )

    res = await run_cvl_generator(
        ctx, g, NatspecGenerationInput(
            curr_spec=None,
            input=[
                f"The current stub implementation of the {contract_name} contract is",
                stub_reader()
            ],
            required_validations=["feedback"],
            skipped=[],
            validations={},
            failed=None,
            suggested_spec_path=None,
            property_rules=[],
        ),
        ctxt=feedback_ctxt,
        description = f"{contract_name} {component.component.name} ({len(props)} properties)"
    )
    assert "result" in res
    assert res["failed"] is not None
    # I hate this, but, well, I give up
    if res["failed"]:
        to_ret = GaveUp(reason=res["result"], ty="fail")
    else:
        assert res["curr_spec"] is not None and res["suggested_spec_path"] is not None
        to_ret = GenerationSuccess(
            commentary=res["result"],
            skipped=res["skipped"],
            spec=res["curr_spec"],
            suggested_path=res["suggested_spec_path"],
            ty="success"
        )
    await root_ctx.cache_put(AuthorResult(result_wrapped=to_ret))
    return to_ret
