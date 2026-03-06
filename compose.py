import composer.certora as _

import argparse
import asyncio
import uuid
from pathlib import Path

from rich.console import Console

from langchain_core.runnables import RunnableConfig

from graphcore.graph import FlowInput

from composer.assistant.agent import build_orchestrator
from composer.assistant.handler import OrchestratorHandler
from composer.assistant.types import OrchestratorContext, OrchestratorModelConfig
from composer.io.ide_bridge import IDEBridge
from composer.io.stream import EventQueue
from composer.rag.db import DEFAULT_CONNECTION as RAG_DEFAULT
from composer.workflow.services import create_llm
from composer.io.graph_runner import run_graph


async def _drain_events(queue: EventQueue, handler: OrchestratorHandler) -> None:
    """Pull events from the queue and render them via the handler."""
    async for event in queue.stream_events():
        handler.on_event(event)


async def main() -> int:
    parser = argparse.ArgumentParser(description="AI-assisted formal verification orchestrator")
    parser.add_argument("--model", default="claude-opus-4-6", help="Model to use")
    parser.add_argument("--tokens", type=int, default=10_000, help="Token budget")
    parser.add_argument("--thinking-tokens", type=int, default=2048, help="Thinking token budget")
    parser.add_argument("--rag-db", default=RAG_DEFAULT, help="RAG database connection string")
    args = parser.parse_args()

    config = OrchestratorModelConfig(
        model=args.model,
        tokens=args.tokens,
        thinking_tokens=args.thinking_tokens,
        memory_tool=True,
        rag_db=args.rag_db,
    )

    llm = create_llm(config)

    # Connect IDE bridge if available
    ide = await IDEBridge.connect()

    # Determine workspace
    workspace: Path
    if ide is not None:
        workspace = await ide.workspace_folder()
    else:
        workspace = Path.cwd()

    console = Console()
    console.print(f"[bold]Workspace:[/bold] {workspace}")

    # Build orchestrator graph
    compiled = build_orchestrator(workspace, llm)
    ctxt = OrchestratorContext(
        workspace=workspace, ide=ide, llm=llm, config=config,
    )

    # Set up handler
    handler = OrchestratorHandler(console=console)

    # Set up event queue for async rendering
    ev_queue = EventQueue(asyncio.Event(), [])
    drainer = asyncio.create_task(_drain_events(ev_queue, handler))

    thread_id = f"orchestrator_{uuid.uuid4().hex[:12]}"
    run_conf: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 50,
    }

    flow_input: FlowInput = {"input": []}

    try:
        final_state = await run_graph(
            event_sink=ev_queue.push,
            graph=compiled,
            ctxt=ctxt,
            input=flow_input,
            run_conf=run_conf,
            description="Orchestrator",
            human_handler=handler.on_interrupt,
        )
    except EOFError:
        console.print("\n[dim]Goodbye.[/dim]")
        return 0
    finally:
        drainer.cancel()
        try:
            await drainer
        except asyncio.CancelledError:
            pass

    if ide is not None:
        await ide.close()

    result = final_state.get("result")
    if result:
        console.print(f"\n[bold]{result}[/bold]")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
