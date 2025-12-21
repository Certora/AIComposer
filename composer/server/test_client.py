import asyncio
import json
import websockets
import argparse
import pathlib

def read_file_as_payload(path: str) -> dict:
    """Helper to read a file and return a payload dictionary."""
    p = pathlib.Path(path)
    return {
        "name": p.name,
        "content": p.read_text()
    }

async def test_client(
    port: int,
    auto_respond: bool = False,
    debug_prompt: str | None = None
) -> None:
    uri = f"ws://localhost:{port}"
    print(f"[CLIENT] Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print(f"[CLIENT] Connected to {uri}")
            # Use trivial example for testing
            config = {
                "interface_file": read_file_as_payload("examples/trivial/Intf.sol"),
                "spec_file": read_file_as_payload("examples/trivial/simple.spec"),
                "system_doc": read_file_as_payload("examples/trivial/system_doc_simple.txt"),
                "debug": True,
                "debug_prompt_override": debug_prompt
            }
            print(f"[CLIENT] Sending start command with file contents...")
            await websocket.send(json.dumps({
                "type": "start",
                "config": config
            }))
            
            print("[CLIENT] Waiting for messages...")
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get("type")
                payload = data.get("payload")
                
                if msg_type == "info":
                    print(f"[CLIENT] [INFO] {payload.get('message')}")
                    if "Workflow finished successfully" in payload.get('message'):
                        print("[CLIENT] Mission accomplished. Closing connection.")
                        break
                elif msg_type == "error":
                    print(f"[CLIENT] [ERROR] {payload.get('message')}")
                elif msg_type == "checkpoint":
                    print(f"[CLIENT] [CHECKPOINT] {payload.get('checkpoint_id')}")
                elif msg_type == "user_update":
                    print(f"[CLIENT] [USER UPDATE] {payload}")
                elif msg_type == "summarize_update":
                    # For summarize update, the payload is the state dict, which can be huge.
                    # Just print a summary of it.
                    nodes = list(payload.keys())
                    print(f"[CLIENT] [SUMMARIZE] Update from nodes: {nodes}")
                elif msg_type == "human_interrupt":
                    print(f"\n[CLIENT] [HUMAN INTERRUPT]")
                    print(f"[CLIENT] Interrupt Type: {payload.get('type')}")
                    if payload.get('type') == 'question':
                        print(f"[CLIENT] Question: {payload.get('question')}")
                    elif payload.get('type') == 'proposal':
                        print(f"[CLIENT] Explanation: {payload.get('explanation')}")
                    
                    # Read multiline input from user
                    if auto_respond:
                        answer = "Please proceed in the way you believe is best."
                        print(f"[CLIENT] Auto-responding: {answer}")
                    else:
                        print("[CLIENT] Enter your answer (double newline to finish):")
                        lines: list[str] = []
                        while True:
                            try:
                                line = input("> ")
                                if line == "" and lines and lines[-1] == "":
                                    break
                                lines.append(line)
                            except EOFError:
                                break
                        answer = "\n".join(lines)
                    
                    print(f"[CLIENT] Sending human response: {answer[:50]}...")
                    await websocket.send(json.dumps({
                        "type": "human_response",
                        "answer": answer
                    }))
                elif msg_type == "thread_id":
                    print(f"[CLIENT] [THREAD ID] {payload.get('thread_id')}")
                else:
                    print(f"[CLIENT] [MSG] {msg_type}: {payload}")
    except Exception as e:
        print(f"[CLIENT] Connection or communication failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8767, help="Port to connect to")
    parser.add_argument("--auto", action="store_true", help="Automatically respond to human interrupts")
    parser.add_argument("--debug-prompt", type=str, help="Override the debug prompt")
    args = parser.parse_args()
    
    try:
        asyncio.run(test_client(args.port, args.auto, args.debug_prompt))
    except KeyboardInterrupt:
        print("\n[CLIENT] Stopped by user")

if __name__ == "__main__":
    main()

