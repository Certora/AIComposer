import * as vscode from 'vscode';

export class DashboardProvider {
    public static readonly viewType = 'certora-ai-composer.dashboard';

    public static createOrShow(extensionUri: vscode.Uri) {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;

        const panel = vscode.window.createWebviewPanel(
            DashboardProvider.viewType,
            'Prover History',
            column || vscode.ViewColumn.One,
            {
                enableScripts: true,
                localResourceRoots: [extensionUri]
            }
        );

        panel.webview.html = this._getHtmlForWebview(panel.webview);
    }

    private static _getHtmlForWebview(webview: vscode.Webview) {
        return `<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Prover History</title>
                <style>
                    body { font-family: sans-serif; padding: 20px; }
                    .run-card { border: 1px solid #ccc; border-radius: 8px; padding: 15px; margin-bottom: 20px; background: #fafafa; }
                    .run-header { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
                    .status-icon { font-size: 20px; }
                    .status-success { color: green; }
                    .status-fail { color: red; }
                    .run-id { font-weight: bold; }
                    .timestamp { color: #666; margin-left: auto; }
                    .metrics { margin-bottom: 10px; }
                    .context { font-style: italic; color: #444; margin-bottom: 15px; }
                    .actions { display: flex; gap: 10px; }
                    button { padding: 5px 10px; cursor: pointer; }
                </style>
            </head>
            <body>
                <h1>Prover History</h1>
                <div id="history-list">
                    <!-- Sample Card 1 -->
                    <div class="run-card">
                        <div class="run-header">
                            <span class="status-icon status-fail">❌</span>
                            <span class="run-id">#124</span>
                            <span class="timestamp">Today, 10:45 AM</span>
                        </div>
                        <div class="metrics">1 Failed, 11 Passed</div>
                        <div class="context">Context: Triggered by Chat: "Verify deposit function"</div>
                        <div class="actions">
                            <button onclick="viewTrace()">View Trace</button>
                            <button onclick="openReport()">Open Run Report ↗</button>
                        </div>
                    </div>

                    <!-- Sample Card 2 -->
                    <div class="run-card">
                        <div class="run-header">
                            <span class="status-icon status-success">✅</span>
                            <span class="run-id">#123</span>
                            <span class="timestamp">Today, 10:00 AM</span>
                        </div>
                        <div class="metrics">12/12 Properties Passed</div>
                        <div class="actions">
                            <button onclick="openReport()">Open Run Report ↗</button>
                        </div>
                    </div>
                </div>
                <script>
                    function viewTrace() {
                        // Notify extension to open trace
                    }
                    function openReport() {
                        // Open external link
                    }
                </script>
            </body>
            </html>`;
    }
}

