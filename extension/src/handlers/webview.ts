import * as vscode from 'vscode';
import { registerMethod } from '../protocol';

const panels = new Map<string, vscode.WebviewPanel>();
let panelCounter = 0;

export function registerWebviewHandlers(): void {
  registerMethod('editor/showWebview', async (params) => {
    const markdown = params.markdown as string;
    const title = (params.title as string | undefined) ?? 'Composer';
    const id = (params.id as string | undefined) ?? `panel-${panelCounter++}`;

    if (!markdown) {
      throw new Error('Missing required param: markdown');
    }

    const rendered = await vscode.commands.executeCommand<string>('markdown.api.render', markdown) ?? '';

    let panel = panels.get(id);
    if (panel) {
      panel.reveal();
    } else {
      panel = vscode.window.createWebviewPanel(
        'composer.webview',
        title,
        vscode.ViewColumn.Beside,
        { enableScripts: false },
      );
      panels.set(id, panel);
      panel.onDidDispose(() => panels.delete(id));
    }

    panel.webview.html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline';">
  <style>
    body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); padding: 16px; }
  </style>
</head>
<body>${rendered}</body>
</html>`;

    return { success: true, panelId: id };
  });
}
