import asyncio
import json
import threading
import websockets
import argparse
from typing import Any, Dict, Optional, List, cast as type_cast

from composer.workflow.executor import execute_ai_composer_workflow
from composer.workflow.factories import create_llm
from composer.input.files import upload_input
from composer.server.ws_io import WebSocketComposerIO
from composer.diagnostics.debug import setup_logging
from composer.input.types import CommandLineArgs, WorkflowOptions

class JsonWorkflowOptions:
    """A helper class to wrap JSON configuration into an object that 
    satisfies the WorkflowOptions/CommandLineArgs protocol.
    """
    prover_capture_output: bool
    prover_keep_folders: bool
    debug_prompt_override: Optional[str]
    checkpoint_id: Optional[str]
    thread_id: Optional[str]
    recursion_limit: int
    audit_db: str
    summarization_threshold: Optional[int]
    requirements_oracle: List[str]
    set_reqs: Optional[str]
    skip_reqs: bool
    rag_db: str
    model: str
    tokens: int
    thinking_tokens: int
    memory_tool: bool
    debug: bool
    interface_file: str
    spec_file: str
    system_doc: str
    debug_fs: Optional[str]

    def __init__(self, data: Dict[str, Any]) -> None:
        # Default values for required fields
        self.prover_capture_output = data.get("prover_capture_output", True)
        self.prover_keep_folders = data.get("prover_keep_folders", False)
        self.debug_prompt_override = data.get("debug_prompt_override")
        self.checkpoint_id = data.get("checkpoint_id")
        self.thread_id = data.get("thread_id")
        self.recursion_limit = data.get("recursion_limit", 50)
        self.audit_db = data.get("audit_db", "postgresql://audit_db_user:audit_db_password@localhost:5432/audit_db")
        self.summarization_threshold = data.get("summarization_threshold")
        self.requirements_oracle = data.get("requirements_oracle", [])
        self.set_reqs = data.get("set_reqs")
        self.skip_reqs = data.get("skip_reqs", False)
        self.rag_db = data.get("rag_db", "postgresql://rag_user:rag_password@localhost:5432/rag_db")
        self.model = data.get("model", "claude-sonnet-4-20250514")
        self.tokens = data.get("tokens", 10000)
        self.thinking_tokens = data.get("thinking_tokens", 2048)
        self.memory_tool = data.get("memory_tool", False)
        self.debug = data.get("debug", False)
        self.interface_file = data.get("interface_file", "")
        self.spec_file = data.get("spec_file", "")
        self.system_doc = data.get("system_doc", "")
        self.debug_fs = data.get("debug_fs")

    def __getattr__(self, name: str) -> Any:
        # Fallback for any missing attributes
        return None

async def handler(websocket: Any) -> None:
    addr = websocket.remote_address
    print(f"[SERVER] New connection from {addr}", flush=True)
    workflow_thread = None
    response_queue: asyncio.Queue = asyncio.Queue()
    io: Optional[WebSocketComposerIO] = None
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                print(f"[SERVER] Received message type: {msg_type} from {addr}", flush=True)
                
                if msg_type == "start":
                    if workflow_thread and workflow_thread.is_alive():
                        print(f"[SERVER] Workflow already running for {addr}", flush=True)
                        await websocket.send(json.dumps({
                            "type": "error",
                            "payload": {"message": "Workflow already running"}
                        }))
                        continue
                    
                    config_data = data.get("config", {})
                    options = JsonWorkflowOptions(config_data)
                    print(f"[SERVER] Starting workflow for {addr} with config keys: {list(config_data.keys())}", flush=True)
                    
                    if not options.interface_file or not options.spec_file or not options.system_doc:
                        print(f"[SERVER] Missing required files for {addr}", flush=True)
                        await websocket.send(json.dumps({
                            "type": "error",
                            "payload": {"message": "Missing required input files (interface_file, spec_file, system_doc)"}
                        }))
                        continue

                    loop = asyncio.get_running_loop()
                    io = WebSocketComposerIO(websocket, loop, response_queue)
                    
                    # Narrow type for the background thread
                    if io is None:
                        continue
                    current_io: WebSocketComposerIO = io
                    
                    def run_workflow() -> None:
                        try:
                            print(f"[WORKFLOW] Initializing for {addr}...", flush=True)
                            setup_logging(options.debug)
                            llm = create_llm(options)
                            input_data = upload_input(type_cast(CommandLineArgs, options), log=current_io.log_info)
                            
                            current_io.log_info("Starting AI Composer workflow...")
                            print(f"[WORKFLOW] Executing workflow for {addr}...", flush=True)
                            execute_ai_composer_workflow(
                                llm=llm,
                                input=input_data,
                                workflow_options=type_cast(WorkflowOptions, options),
                                io=current_io
                            )
                            current_io.log_info("Workflow finished successfully")
                            print(f"[WORKFLOW] Finished successfully for {addr}", flush=True)
                        except Exception as e:
                            print(f"[WORKFLOW] Failed for {addr}: {str(e)}", flush=True)
                            current_io.log_error(f"Workflow failed: {str(e)}")
                            import traceback
                            traceback.print_exc()

                    workflow_thread = threading.Thread(target=run_workflow, daemon=True)
                    workflow_thread.start()
                    print(f"[SERVER] Workflow thread started for {addr}", flush=True)
                    
                elif msg_type == "interrupt":
                    print(f"[SERVER] Received interrupt request from {addr}", flush=True)
                    if io:
                        io.interrupt_requested = True
                    
                elif msg_type == "human_response":
                    print(f"[SERVER] Received human response from {addr}", flush=True)
                    await response_queue.put(data)

            except Exception as e:
                error_msg = f"Error processing message: {str(e)}"
                print(f"[SERVER] {error_msg} from {addr}", flush=True)
                import traceback
                traceback.print_exc()
                try:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "payload": {"message": f"server error: {error_msg}"}
                    }))
                except:
                    pass
    except Exception as e:
        print(f"[SERVER] Connection error with {addr}: {e}", flush=True)
    finally:
        print(f"[SERVER] Connection closed for {addr}", flush=True)

async def start_server(host: str, port: int) -> None:
    async with websockets.serve(handler, host, port):
        print(f"AI Composer WebSocket server started on ws://{host}:{port} (RELOADED)", flush=True)
        await asyncio.Future()  # run forever

def main() -> None:
    parser = argparse.ArgumentParser(description="AI Composer WebSocket Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    args = parser.parse_args()
    
    try:
        asyncio.run(start_server(args.host, args.port))
    except KeyboardInterrupt:
        print("\nServer stopped")

if __name__ == "__main__":
    main()
