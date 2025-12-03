import bind as _

import sys
import pprint

from langgraph.checkpoint.base import CheckpointTuple
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage

from composer.workflow.factories import get_checkpointer

thread_id = sys.argv[1]

checkpointer = get_checkpointer()
r = checkpointer.get_tuple({
    "configurable": {
        "thread_id": thread_id
    }
})

if r is None:
    print(f"No data found for {thread_id}")
    sys.exit(1)

def search_loop(tup: CheckpointTuple):
    print(f"Checkpoint id {tup.checkpoint['id']}")
    messages = tup.checkpoint["channel_values"]["messages"]
    print("Last message:")
    m = messages[-1]
    match m:
        case AIMessage():
            print("AI Message")
            print("Text: ")
            print(m.text())
            for t in m.tool_calls:
                pprint.pprint(t)
        case ToolMessage():
            print("Tool message")
            print("Text: ")
            print(m.text())
        case HumanMessage() | SystemMessage():
            print("Human/System message:")
            print(m.text())
        case _:
            print(f"Unhandled message: {type(m)}")
    
    while True:
        d = input("What next? p for previous checkpoint, n for next ").strip()
        match d:
            case 'p':
                parent = tup.parent_config
                if parent is None:
                    print("No parent")
                    continue
                parent_tup = checkpointer.get_tuple(parent)
                if parent_tup is None:
                    print("FATAL: parent tuple is not in the db?")
                    continue
                search_loop(parent_tup)
                search_loop(tup)
            case 'n':
                return
            case _:
                print("Unrecognized command")

search_loop(r)


