import logging
import pathlib

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from verisafe.input.types import CommandLineArgs
from verisafe.workflow.factories import get_checkpointer, get_cryptostate_builder


def setup_logging(debug: bool) -> None:
    """Configure logging based on debug flag."""
    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

def dump_fs(args: CommandLineArgs, llm: BaseChatModel) -> int:
    workflow = get_cryptostate_builder(
        llm=llm,
        fs_layer=None,
        summarization_threshold=None
    )[0]
    config: RunnableConfig = {
        "configurable": {
            "thread_id": args.thread_id,
            "checkpoint_id": args.checkpoint_id
        }
    }
    build = workflow.compile(checkpointer=get_checkpointer())
    st = build.get_state(config)
    output = pathlib.Path(args.debug_fs)
    output.mkdir(exist_ok=True, parents=True)
    for (t, r) in st.values["vfs"].items():
        out_path : pathlib.Path = output / t
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(r)
    return 0