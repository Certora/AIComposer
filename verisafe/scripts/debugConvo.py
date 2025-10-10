import bind as _

import pathlib
import sys

from langchain_core.messages import HumanMessage
from dataclasses import dataclass

if __name__ != "__main__":
    raise RuntimeError("This is a script only module")

verisafe_dir = str(pathlib.Path(__file__).parent.parent.parent.absolute())

if verisafe_dir not in sys.path:
    sys.path.append(verisafe_dir)

import verisafe.certora as _
from verisafe.workflow.factories import get_checkpointer, create_llm

checkpoint = get_checkpointer()

thread_id = sys.argv[1]
checkpoint_id = sys.argv[2]

msgs = checkpoint.get_tuple({
    "configurable": {
        "thread_id": thread_id,
        "checkpoint_id": checkpoint_id
    }
}).checkpoint["channel_values"]["messages"].copy() #type: ignore

msgs.append(HumanMessage(
    content=sys.argv[3]
))

@dataclass
class ModelOpts:
    model: str
    tokens: int
    thinking_tokens: int

opts = ModelOpts(
    model="claude-sonnet-4-5-20250929",
    thinking_tokens=2048,
    tokens=4096
)

llm = create_llm(opts)
resp = llm.invoke(msgs)
print(resp.text())