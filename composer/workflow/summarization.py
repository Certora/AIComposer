from langgraph.config import get_stream_writer

from graphcore.summary import SummaryConfig

from composer.diagnostics.stream import SummarizationPartial
from composer.core.state import AIComposerState


class SummaryGeneration(SummaryConfig[AIComposerState]):
    def get_resume_prompt(self, state: AIComposerState, summary: str) -> str:
        res = super().get_resume_prompt(state, summary)

        res += "\n You may use the VFS tools to query the current state of your implementation."
        return res
    
    def get_summarization_prompt(self, state: AIComposerState) -> str:
        return super().get_summarization_prompt(state) + "\n In addition, if you have a TODO " \
            "list from prior execution of the prover, retain that TODO list in your summary. \n" \
            "You should also preserve the natural language list of implementation requirements, including" \
            "their order."

    def on_summary(self, state: AIComposerState, summary: str, resume: str) -> None:
        writer = get_stream_writer()
        summ : SummarizationPartial = {
            "type": "summarization_raw",
            "summary": summary
        }
        writer(summ)