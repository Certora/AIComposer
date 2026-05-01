import logging
import pathlib
from typing import Any

from langchain_core.runnables import RunnableConfig

from composer.workflow.services import checkpointer_context


def setup_logging(debug: bool) -> None:
    """Configure logging based on debug flag."""
    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

async def dump_fs(debug_fs: str, *, thread_id : str | None, checkpoint_id: str | None) -> int:
    configurable : dict[str, Any] = {}
    if thread_id:
        configurable["thread_id"] = thread_id
    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id
    config: RunnableConfig = {
        "configurable": configurable
    }
    async with checkpointer_context() as check:
        tup = await check.aget_tuple(config)
    if tup is None:
        print("Invalid config, no state found")
        return 1
    
    output = pathlib.Path(debug_fs)
    output.mkdir(exist_ok=True, parents=True)
    for (t, r) in tup.checkpoint["channel_values"]["vfs"].items():
        out_path : pathlib.Path = output / t
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(r)
    return 0