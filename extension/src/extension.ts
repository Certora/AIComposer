import * as vscode from 'vscode';
import { startServer, stopServer } from './server';
import { VirtualDocProvider } from './providers/virtualDoc';
import { ProposedTreeProvider } from './providers/proposedTree';
import { registerWorkspaceHandlers } from './handlers/workspace';
import { registerEditorHandlers } from './handlers/editor';
import { registerWebviewHandlers } from './handlers/webview';
import { registerResultsHandlers, registerResultsCommands } from './handlers/results';

export function activate(context: vscode.ExtensionContext): void {
  const virtualDocProvider = new VirtualDocProvider();
  const proposedTree = new ProposedTreeProvider();

  // Register providers
  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider('composer', virtualDocProvider),
    vscode.window.createTreeView('composerProposedChanges', { treeDataProvider: proposedTree }),
    virtualDocProvider,
    proposedTree,
  );

  // Register handlers
  registerWorkspaceHandlers(virtualDocProvider);
  registerEditorHandlers(virtualDocProvider);
  registerWebviewHandlers();
  registerResultsHandlers(proposedTree, virtualDocProvider);
  context.subscriptions.push(...registerResultsCommands(proposedTree, virtualDocProvider));

  // Start WebSocket server and inject env vars into terminal
  startServer((info) => {
    const env = context.environmentVariableCollection;
    env.replace('COMPOSER_WS_PORT', String(info.port));
    env.replace('COMPOSER_AUTH_TOKEN', info.token);

    vscode.window.showInformationMessage(`Composer bridge listening on port ${info.port}`);
  });
}

export function deactivate(): void {
  stopServer();
}
