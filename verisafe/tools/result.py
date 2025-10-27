from verisafe.core.state import ResultStateSchema
from graphcore.tools.results import result_tool_generator

code_result = result_tool_generator(
    "generated_code",
    ResultStateSchema,
    doc="""
    Used to communicate when the generated code is complete and satisfies all of the rules in specification.
    """
)
