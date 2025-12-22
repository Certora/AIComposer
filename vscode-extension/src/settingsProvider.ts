import * as vscode from 'vscode';

export class SettingsProvider {
    public static readonly viewType = 'certora-ai-composer.settings';
    public static currentPanel: SettingsProvider | undefined;

    private readonly _panel: vscode.WebviewPanel;
    private readonly _extensionUri: vscode.Uri;
    private _disposables: vscode.Disposable[] = [];

    private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri) {
        this._panel = panel;
        this._extensionUri = extensionUri;

        this._panel.webview.html = SettingsProvider._getHtmlForWebview(this._panel.webview, this._extensionUri);

        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);

        this._panel.webview.onDidReceiveMessage(
            async (message) => {
                switch (message.command) {
                    case 'save':
                        await SettingsProvider._saveSettings(message.settings);
                        vscode.window.showInformationMessage('Settings saved successfully');
                        break;
                    case 'reset':
                        await SettingsProvider._resetSettings();
                        this._panel.webview.html = SettingsProvider._getHtmlForWebview(this._panel.webview, this._extensionUri);
                        vscode.window.showInformationMessage('Settings reset to defaults');
                        break;
                }
            },
            null,
            this._disposables
        );
    }

    public static createOrShow(extensionUri: vscode.Uri) {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;

        if (SettingsProvider.currentPanel) {
            SettingsProvider.currentPanel._panel.reveal(column);
            return;
        }

        const panel = vscode.window.createWebviewPanel(
            SettingsProvider.viewType,
            'AI Composer Settings',
            column || vscode.ViewColumn.One,
            {
                enableScripts: true,
                localResourceRoots: [extensionUri],
                retainContextWhenHidden: true
            }
        );

        SettingsProvider.currentPanel = new SettingsProvider(panel, extensionUri);
    }

    public dispose() {
        SettingsProvider.currentPanel = undefined;

        this._panel.dispose();

        while (this._disposables.length) {
            const x = this._disposables.pop();
            if (x) {
                x.dispose();
            }
        }
    }

    private static async _saveSettings(settings: any) {
        const config = vscode.workspace.getConfiguration('certora-ai-composer');
        for (const [key, value] of Object.entries(settings)) {
            await config.update(key, value, vscode.ConfigurationTarget.Global);
        }
    }

    private static async _resetSettings() {
        const config = vscode.workspace.getConfiguration('certora-ai-composer');
        await config.update('serverPort', undefined, vscode.ConfigurationTarget.Global);
        await config.update('model', undefined, vscode.ConfigurationTarget.Global);
        await config.update('memoryTool', undefined, vscode.ConfigurationTarget.Global);
        await config.update('debugPromptOverride', undefined, vscode.ConfigurationTarget.Global);
    }

    private static _getHtmlForWebview(webview: vscode.Webview, extensionUri: vscode.Uri) {
        const config = vscode.workspace.getConfiguration('certora-ai-composer');
        const serverPort = config.get<number>('serverPort') || 8769;
        const model = config.get<string>('model') || 'claude-sonnet-4-20250514';
        const memoryTool = config.get<boolean>('memoryTool') || false;
        const debugPromptOverride = config.get<string>('debugPromptOverride') || '';

        return `<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>AI Composer Settings</title>
                <style>
                    body {
                        font-family: var(--vscode-font-family);
                        color: var(--vscode-foreground);
                        background-color: var(--vscode-editor-background);
                        padding: 40px;
                        max-width: 800px;
                        margin: 0 auto;
                    }
                    .header {
                        display: flex;
                        align-items: center;
                        gap: 15px;
                        margin-bottom: 40px;
                        border-bottom: 1px solid var(--vscode-widget-border);
                        padding-bottom: 20px;
                    }
                    .header h1 {
                        margin: 0;
                        font-size: 24px;
                        font-weight: 500;
                    }
                    .section {
                        margin-bottom: 30px;
                    }
                    .section-title {
                        font-size: 16px;
                        font-weight: 600;
                        margin-bottom: 15px;
                        color: var(--vscode-settings-headerForeground);
                    }
                    .setting-item {
                        display: flex;
                        flex-direction: column;
                        gap: 8px;
                        margin-bottom: 20px;
                        padding: 15px;
                        background: var(--vscode-settings-cardBackground, rgba(255,255,255,0.03));
                        border-radius: 8px;
                    }
                    .setting-header {
                        display: flex;
                        justify-content: space-between;
                        align-items: flex-start;
                    }
                    .setting-info {
                        flex: 1;
                    }
                    .setting-label {
                        font-weight: 500;
                        font-size: 13px;
                    }
                    .setting-description {
                        font-size: 12px;
                        color: var(--vscode-descriptionForeground);
                        margin-top: 4px;
                    }
                    .setting-control {
                        margin-top: 10px;
                    }
                    input[type="text"], input[type="number"], select {
                        width: 100%;
                        max-width: 400px;
                        padding: 8px;
                        background: var(--vscode-input-background);
                        color: var(--vscode-input-foreground);
                        border: 1px solid var(--vscode-input-border);
                        border-radius: 4px;
                        outline: none;
                    }
                    input[type="checkbox"] {
                        width: 18px;
                        height: 18px;
                        cursor: pointer;
                    }
                    input:focus {
                        border-color: var(--vscode-focusBorder);
                    }
                    .footer {
                        margin-top: 50px;
                        display: flex;
                        gap: 15px;
                        position: sticky;
                        bottom: 0;
                        background: var(--vscode-editor-background);
                        padding: 20px 0;
                        border-top: 1px solid var(--vscode-widget-border);
                    }
                    button {
                        padding: 8px 20px;
                        border-radius: 4px;
                        border: none;
                        cursor: pointer;
                        font-weight: 500;
                    }
                    .btn-primary {
                        background: var(--vscode-button-background);
                        color: var(--vscode-button-foreground);
                    }
                    .btn-primary:hover {
                        background: var(--vscode-button-hoverBackground);
                    }
                    .btn-secondary {
                        background: var(--vscode-button-secondaryBackground);
                        color: var(--vscode-button-secondaryForeground);
                    }
                    .btn-secondary:hover {
                        background: var(--vscode-button-secondaryHoverBackground);
                    }
                </style>
            </head>
            <body>
                <div class="header">
                    <span style="font-size: 32px;">⚙️</span>
                    <h1>AI Composer Settings</h1>
                </div>

                <div class="section">
                    <div class="section-title">General</div>
                    <div class="setting-item">
                        <div class="setting-info">
                            <div class="setting-label">AI Model</div>
                            <div class="setting-description">The AI model to use for composer tasks. Default is Claude 4 Sonnet.</div>
                        </div>
                        <div class="setting-control">
                            <select id="model">
                                <option value="claude-sonnet-4-20250514" ${model === 'claude-sonnet-4-20250514' ? 'selected' : ''}>Claude 4 Sonnet</option>
                                <option value="claude-3-5-sonnet-20240620" ${model === 'claude-3-5-sonnet-20240620' ? 'selected' : ''}>Claude 3.5 Sonnet</option>
                                <option value="claude-3-opus-20240229" ${model === 'claude-3-opus-20240229' ? 'selected' : ''}>Claude 3 Opus</option>
                                <option value="claude-3-haiku-20240307" ${model === 'claude-3-haiku-20240307' ? 'selected' : ''}>Claude 3 Haiku</option>
                            </select>
                        </div>
                    </div>

                    <div class="setting-item">
                        <div class="setting-header">
                            <div class="setting-info">
                                <div class="setting-label">Enable Memory Tool</div>
                                <div class="setting-description">Allow the AI composer to use memory tools during reasoning.</div>
                            </div>
                            <div class="setting-control" style="margin-top: 0;">
                                <input type="checkbox" id="memoryTool" ${memoryTool ? 'checked' : ''} />
                            </div>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">Connection</div>
                    <div class="setting-item">
                        <div class="setting-info">
                            <div class="setting-label">Server Port</div>
                            <div class="setting-description">The port the AI Composer WebSocket server is listening on. Default is 8769.</div>
                        </div>
                        <div class="setting-control">
                            <input type="number" id="serverPort" value="${serverPort}" />
                        </div>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">Debug</div>
                    <div class="setting-item">
                        <div class="setting-info">
                            <div class="setting-label">Debug Prompt Override</div>
                            <div class="setting-description">Manually override the prompt sent to the AI for debugging purposes.</div>
                        </div>
                        <div class="setting-control">
                            <input type="text" id="debugPromptOverride" value="${debugPromptOverride}" placeholder="Empty uses default chat message" />
                        </div>
                    </div>
                </div>

                <div class="footer">
                    <button class="btn-primary" onclick="save()">Save Changes</button>
                    <button class="btn-secondary" onclick="reset()">Reset to Defaults</button>
                </div>

                <script>
                    const vscode = acquireVsCodeApi();

                    function save() {
                        const settings = {
                            serverPort: parseInt(document.getElementById('serverPort').value),
                            model: document.getElementById('model').value,
                            memoryTool: document.getElementById('memoryTool').checked,
                            debugPromptOverride: document.getElementById('debugPromptOverride').value
                        };
                        vscode.postMessage({ command: 'save', settings });
                    }

                    function reset() {
                        vscode.postMessage({ command: 'reset' });
                    }
                </script>
            </body>
            </html>`;
    }
}
