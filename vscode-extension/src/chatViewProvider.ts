import * as vscode from 'vscode';
import { WebSocket } from 'ws';
import { ContextManagerProvider } from './contextManagerProvider';

export class ChatViewProvider implements vscode.WebviewViewProvider {
    private _view?: vscode.WebviewView;
    private _socket?: WebSocket;

    constructor(
        private readonly _extensionUri: vscode.Uri,
        private readonly _contextManager: ContextManagerProvider
    ) {}

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(async data => {
            try {
                switch (data.type) {
                    case 'sendMessage':
                        await this._handleSendMessage(data.value);
                        break;
                    case 'connect':
                        this._connectToServer();
                        break;
                case 'humanResponse':
                    this._handleHumanResponse(data.value);
                    break;
                case 'openSettings':
                    vscode.commands.executeCommand('certora-ai-composer.openSettings');
                    break;
                case 'webviewError':
                        console.error(`[Webview Error] ${data.value}`);
                        vscode.window.showErrorMessage(`AI Composer UI Error: ${data.value}`);
                        break;
                }
            } catch (err: any) {
                console.error(`[Extension Error] ${err.message}`);
                vscode.window.showErrorMessage(`AI Composer Extension Error: ${err.message}`);
                this._view?.webview.postMessage({ type: 'error', value: err.message });
            }
        });
    }

    private _handleHumanResponse(answer: string) {
        if (this._socket && this._socket.readyState === WebSocket.OPEN) {
            this._socket.send(JSON.stringify({
                type: 'human_response',
                answer: answer
            }));
        }
    }

    private _connectToServer() {
        if (this._socket) {
            this._socket.close();
        }

        const config = vscode.workspace.getConfiguration('certora-ai-composer');
        const port = config.get<number>('serverPort') || 8769;

        console.log(`[ChatViewProvider] Connecting to ws://localhost:${port}`);
        this._socket = new WebSocket(`ws://localhost:${port}`);

        this._socket.on('open', () => {
            console.log('[ChatViewProvider] Connection established');
            this._view?.webview.postMessage({ type: 'status', value: 'Connected' });
        });

        this._socket.on('message', (data: any) => {
            const message = JSON.parse(data.toString());
            this._handleServerMessage(message);
        });

        this._socket.on('close', () => {
            console.log('[ChatViewProvider] Connection closed');
            this._view?.webview.postMessage({ type: 'status', value: 'Disconnected' });
        });

        this._socket.on('error', (err: any) => {
            console.error('[ChatViewProvider] WebSocket error:', err);
            this._view?.webview.postMessage({ type: 'error', value: `Connection failed: ${err.message}` });
        });
    }

    private async _handleSendMessage(data: { text: string }) {
        if (this._socket && this._socket.readyState === WebSocket.OPEN) {
            const context = this._contextManager.getContext();
            const config = vscode.workspace.getConfiguration('certora-ai-composer');
            
            // Validation: Check if required context files are provided
            if (!context.interfaces.length || !context.specs.length || !context.docs.length) {
                const missing = [];
                if (!context.interfaces.length) missing.push("an Interface file");
                if (!context.specs.length) missing.push("a Spec file");
                if (!context.docs.length) missing.push("a System Doc");
                
                this._view?.webview.postMessage({ 
                    type: 'serverMessage', 
                    value: { 
                        type: 'error', 
                        payload: { message: `Please add ${missing.join(', ')} to the Context Manager before starting.` } 
                    } 
                });
                return;
            }

            try {
                const readFile = async (path: string) => {
                    try {
                        const uri = vscode.Uri.file(path);
                        const content = await vscode.workspace.fs.readFile(uri);
                        return {
                            name: path.split('/').pop() || path,
                            content: new TextDecoder().decode(content)
                        };
                    } catch (e: any) {
                        throw new Error(`Could not read file at ${path}. Please ensure it exists and has not been moved.`);
                    }
                };

                const startMsg = {
                    type: 'start',
                    config: {
                        interface_file: await readFile(context.interfaces[0]),
                        spec_file: await readFile(context.specs[0]),
                        system_doc: await readFile(context.docs[0]),
                        debug_prompt_override: config.get<string>('debugPromptOverride') || data.text,
                        model: config.get<string>('model') || 'claude-sonnet-4-20250514',
                        memory_tool: config.get<boolean>('memoryTool') || false
                    }
                };
                this._socket.send(JSON.stringify(startMsg));
            } catch (err: any) {
                this._view?.webview.postMessage({ 
                    type: 'error', 
                    value: `Failed to read context files: ${err.message}` 
                });
            }
        } else {
            this._view?.webview.postMessage({ type: 'error', value: 'Not connected to server' });
        }
    }

    private _handleServerMessage(message: any) {
        this._view?.webview.postMessage({ type: 'serverMessage', value: message });
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        const logoUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'resources', 'icon.png'));

        return `<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>AI Composer Chat</title>
                <style>
                    body { font-family: var(--vscode-font-family); color: var(--vscode-editor-foreground); background-color: var(--vscode-editor-background); padding: 0 10px 10px 10px; display: flex; flex-direction: column; height: 100vh; box-sizing: border-box; }
                    
                    .sidebar-header {
                        padding: 10px 0;
                        text-transform: uppercase;
                        font-size: 11px;
                        font-weight: bold;
                        color: var(--vscode-descriptionForeground);
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    }
                    .header-logo { width: 14px; height: 14px; }
                    .header-title { flex: 1; }
                    .header-gear { cursor: pointer; font-size: 14px; }

                    #chat-stream { flex: 1; overflow-y: auto; margin-bottom: 10px; padding: 5px; }
                    #input-container { display: flex; gap: 5px; background: var(--vscode-input-background); border: 1px solid var(--vscode-input-border); border-radius: 4px; padding: 5px; }
                    #user-input { flex: 1; background: transparent; color: var(--vscode-input-foreground); border: none; outline: none; }
                    .message { margin-bottom: 15px; padding: 10px; border-radius: 8px; line-height: 1.4; font-size: 13px; }
                    .user { background: var(--vscode-editor-inactiveSelectionBackground); color: var(--vscode-editor-foreground); align-self: flex-end; margin-left: 20px; border: 1px solid var(--vscode-widget-border); }
                    .ai { background: var(--vscode-editor-inactiveSelectionBackground); align-self: flex-start; margin-right: 20px; }
                    
                    .primary-button { 
                        background: #28a745; 
                        color: white; 
                        padding: 12px 24px; 
                        border-radius: 8px; 
                        border: none; 
                        cursor: pointer; 
                        font-weight: bold; 
                        font-size: 14px;
                        transition: background 0.3s, opacity 0.3s;
                    }
                    .primary-button:hover:not(:disabled) { background: #218838; }
                    .primary-button:disabled { 
                        background: var(--vscode-button-secondaryBackground); 
                        color: var(--vscode-button-secondaryForeground); 
                        cursor: not-allowed; 
                        opacity: 0.6; 
                    }
                    
                    .thinking-container { margin-bottom: 10px; }
                    .thinking-header { cursor: pointer; display: flex; align-items: center; gap: 5px; font-weight: bold; color: var(--vscode-descriptionForeground); font-size: 0.9em; }
                    .thinking-logs { display: block; margin-top: 5px; padding: 8px; background: var(--vscode-textCodeBlock-background); border-radius: 4px; font-family: var(--vscode-editor-font-family); font-size: 0.85em; white-space: pre-wrap; max-height: 12em; overflow-y: scroll; scroll-behavior: smooth; }
                    .thinking-container.expanded .thinking-logs { display: block; }
                    .thinking-container.expanded .thinking-icon { transform: rotate(90deg); }
                    .thinking-icon { transition: transform 0.2s; }

                    .human-interrupt { border: 1px solid var(--vscode-editor-inactiveSelectionBackground); background: var(--vscode-editor-background); padding: 15px; border-radius: 4px; margin: 10px 0; }
                    .human-interrupt-question { font-weight: bold; margin-bottom: 10px; color: var(--vscode-editor-foreground); }
                    .interrupt-input-row { display: flex; flex-direction: column; gap: 8px; }
                    .interrupt-input { width: 100%; background: var(--vscode-input-background); color: var(--vscode-input-foreground); border: 1px solid var(--vscode-input-border); padding: 8px; border-radius: 2px; box-sizing: border-box; resize: none; font-family: inherit; }
                    .interrupt-submit { background: var(--vscode-button-background); color: var(--vscode-button-foreground); border: none; padding: 6px 20px; border-radius: 2px; cursor: pointer; align-self: center; margin-top: 5px; }
                    
                    #status-bar { font-size: 0.8em; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
                    .status-connected { color: #4caf50; }
                    .status-disconnected { color: #f44336; }
                    button.secondary { background: var(--vscode-button-secondaryBackground); color: var(--vscode-button-secondaryForeground); border: none; padding: 2px 6px; border-radius: 2px; cursor: pointer; }
                </style>
            </head>
            <body>
                <div id="status-bar">
                    <span id="status-text" class="status-disconnected">Disconnected</span>
                    <button class="secondary" id="connect-btn" onclick="connect()">Connect</button>
                </div>

                <div id="chat-stream">
                    <!-- Initial Action Button from Mockup -->
                    <div id="initial-actions" style="display: flex; justify-content: center; margin: 20px 0;">
                        <button id="generate-btn" class="primary-button" onclick="startInitialWorkflow()">
                            Generate Verified Contract
                        </button>
                    </div>
                </div>
                <!-- Chat input is hidden by default and only revealed for questions if needed. 
                     However, per design, we use inline blocks for human-in-the-loop. -->
                <div id="input-container" style="display: none;">
                    <input type="text" id="user-input" placeholder="Ask AI Composer..." onkeydown="if(event.key==='Enter') send()" />
                    <button class="secondary" onclick="send()">Send</button>
                </div>

                <script>
                    const vscode = acquireVsCodeApi();

                    window.onerror = function(message, source, lineno, colno, error) {
                        const errorInfo = message + ' at ' + source + ':' + lineno + ':' + colno;
                        vscode.postMessage({ type: 'webviewError', value: errorInfo });
                        return true;
                    };

                    const chatStream = document.getElementById('chat-stream');
                    const userInput = document.getElementById('user-input');
                    const statusText = document.getElementById('status-text');
                    const connectBtn = document.getElementById('connect-btn');
                    const generateBtn = document.getElementById('generate-btn');

                    let currentLogs = null;
                    let isWaitingForResponse = false;
                    let isCodeGenerationCompleted = false;
                    let codeCompletionBubble = null;
                    let codeCompletionBuffer = "";

                    function openSettings() {
                        vscode.postMessage({ type: 'openSettings' });
                    }

                    function connect() {
                        try {
                            vscode.postMessage({ type: 'connect' });
                        } catch (e) {
                            window.onerror(e.message, 'chatViewProvider.ts', 0, 0, e);
                        }
                    }

                    function startInitialWorkflow() {
                        try {
                            if (generateBtn.disabled) return;
                            generateBtn.disabled = true;
                            generateBtn.textContent = 'Running...';
                            vscode.postMessage({ type: 'sendMessage', value: { text: "Generate Verified Contract" } });
                        } catch (e) {
                            generateBtn.disabled = false;
                            generateBtn.textContent = 'Generate Verified Contract';
                            window.onerror(e.message, 'chatViewProvider.ts', 0, 0, e);
                        }
                    }

                    function send() {
                        try {
                            const text = userInput.value;
                            if (text && isWaitingForResponse) {
                                addMessage('user', text);
                                vscode.postMessage({ type: 'humanResponse', value: text });
                                isWaitingForResponse = false;
                                document.getElementById('input-container').style.display = 'none';
                                userInput.value = '';
                            }
                        } catch (e) {
                            window.onerror(e.message, 'chatViewProvider.ts', 0, 0, e);
                        }
                    }

                    function addMessage(role, text, isHtml = false) {
                        const div = document.createElement('div');
                        div.className = 'message ' + role;
                        if (isHtml) div.innerHTML = text;
                        else div.textContent = text;
                        chatStream.appendChild(div);
                        chatStream.scrollTop = chatStream.scrollHeight;
                        return div;
                    }

                    function createThinkingBlock() {
                        const container = document.createElement('div');
                        container.className = 'thinking-container expanded';
                        container.innerHTML = \`
                            <div class="thinking-header" onclick="this.parentElement.classList.toggle('expanded')">
                                <span class="thinking-icon">▶</span>
                                <span>Thinking...</span>
                            </div>
                            <div class="thinking-logs"></div>
                        \`;
                        chatStream.appendChild(container);
                        chatStream.scrollTop = chatStream.scrollHeight;
                        return container.querySelector('.thinking-logs');
                    }

                    function createInterruptBlock(question) {
                        const div = document.createElement('div');
                        div.className = 'human-interrupt';
                        div.innerHTML = \`
                            <div class="human-interrupt-question">Question</div>
                            <div style="font-size: 0.9em; margin-bottom: 10px; color: var(--vscode-descriptionForeground);">\${question}</div>
                        \`;
                        chatStream.appendChild(div);
                        chatStream.scrollTop = chatStream.scrollHeight;
                        document.getElementById('input-container').style.display = 'flex';
                        userInput.focus();
                    }

                    function handleServerMessage(msg) {
                        switch (msg.type) {
                            case 'info':
                                const text = msg.payload.message;
                                if (text.includes("= CODE GENERATION COMPLETED =")) {
                                    isCodeGenerationCompleted = true;
                                    currentLogs = null;
                                }
                                if (isCodeGenerationCompleted) {
                                    codeCompletionBuffer += text + "\\n";
                                    if (!codeCompletionBubble) codeCompletionBubble = addMessage('ai', '', true);
                                    let formatted = codeCompletionBuffer
                                        .replace(/= CODE GENERATION COMPLETED =/g, '<strong>$0</strong><br>')
                                        .replace(/Generated Source Files:/g, '<strong>$0</strong><br>')
                                        .replace(/--- (.*?) ---/g, '<div style="margin-top:10px; font-weight:bold; color:var(--vscode-textLink-foreground)">📄 $1</div><pre style="background:var(--vscode-editor-background); padding:8px; border-radius:4px; overflow-x:auto; border:1px solid var(--vscode-widget-border); font-size:12px; line-height:1.2; margin-top:4px"><code>')
                                        .split('<div style="margin-top:10px').map((part, i, arr) => {
                                            if (i === 0) return part;
                                            return '<div style="margin-top:10px' + part + (i < arr.length - 1 ? '</code></pre>' : '');
                                        }).join('');
                                    if (formatted.includes('<code>') && !formatted.endsWith('</code></pre>')) formatted += '</code></pre>';
                                    codeCompletionBubble.innerHTML = formatted;
                                    chatStream.scrollTop = chatStream.scrollHeight;
                                } else {
                                    if (!currentLogs) currentLogs = createThinkingBlock();
                                    currentLogs.textContent += text + '\\n';
                                    currentLogs.scrollTop = currentLogs.scrollHeight;
                                }
                                if (text.includes("Workflow finished successfully")) {
                                    generateBtn.disabled = false;
                                    generateBtn.textContent = 'Generate Verified Contract';
                                    isCodeGenerationCompleted = false;
                                    codeCompletionBubble = null;
                                    codeCompletionBuffer = "";
                                }
                                break;
                            case 'human_interrupt':
                                currentLogs = null;
                                isWaitingForResponse = true;
                                createInterruptBlock(msg.payload.question || 'Please provide clarification:');
                                break;
                            case 'error':
                                generateBtn.disabled = false;
                                generateBtn.textContent = 'Generate Verified Contract';
                                addMessage('ai', 'Error: ' + msg.payload.message);
                                break;
                            case 'checkpoint':
                                if (!currentLogs) currentLogs = createThinkingBlock();
                                currentLogs.textContent += 'Reached checkpoint: ' + msg.payload.checkpoint_id + '\\n';
                                currentLogs.scrollTop = currentLogs.scrollHeight;
                                break;
                        }
                    }

                    window.addEventListener('message', event => {
                        const message = event.data;
                        switch (message.type) {
                            case 'status':
                                statusText.textContent = message.value;
                                statusText.className = 'status-' + message.value.toLowerCase();
                                connectBtn.textContent = message.value === 'Connected' ? 'Disconnect' : 'Connect';
                                break;
                            case 'error':
                                if (message.value.includes('Not connected') || message.value.includes('Connection failed')) {
                                    generateBtn.disabled = false;
                                    generateBtn.textContent = 'Generate Verified Contract';
                                }
                                addMessage('ai', 'Error: ' + message.value);
                                break;
                            case 'serverMessage':
                                handleServerMessage(message.value);
                                break;
                        }
                    });
                </script>
            </body>
            </html>`;
    }
}

