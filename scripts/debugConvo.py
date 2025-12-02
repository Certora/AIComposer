import bind as _

import sys

from langchain_core.messages import HumanMessage
from dataclasses import dataclass

if __name__ != "__main__":
    raise RuntimeError("This is a script only module")

from composer.workflow.factories import get_checkpointer, create_llm

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
    memory_tool: bool

opts = ModelOpts(
    model="claude-sonnet-4-5-20250929",
    thinking_tokens=2048,
    tokens=4096,
    memory_tool=False
)

llm = create_llm(opts)
resp = llm.invoke(msgs)
print(resp.text())