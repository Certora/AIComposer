import * as vscode from 'vscode';
import { startServer, stopServer } from './server';
import { Diagnostics } from './diagnostics';
import { VirtualDocProvider } from './providers/virtualDoc';
import { ProposedTreeProvider } from './providers/proposedTree';
import { registerWorkspaceHandlers } from './handlers/workspace';
import { registerEditorHandlers } from './handlers/editor';
import { registerWebviewHandlers } from './handlers/webview';
import { registerResultsHandlers, registerResultsCommands } from './handlers/results';

let diagnostics: Diagnostics | null = null;

export function activate(context: vscode.ExtensionContext): void {
  diagnostics = new Diagnostics();
  context.subscriptions.push(diagnostics);

  context.subscriptions.push(
    vscode.commands.registerCommand('composer.bridge.status', () => {
      diagnostics?.showStatus();
    }),
  );

  const virtualDocProvider = new VirtualDocProvider();
  const proposedTree = new ProposedTreeProvider();

  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider('composer', virtualDocProvider),
    vscode.window.createTreeView('composerProposedChanges', { treeDataProvider: proposedTree }),
    virtualDocProvider,
    proposedTree,
  );

  registerWorkspaceHandlers(virtualDocProvider);
  registerEditorHandlers(virtualDocProvider);
  registerWebviewHandlers();
  registerResultsHandlers(proposedTree, virtualDocProvider);
  context.subscriptions.push(...registerResultsCommands(proposedTree, virtualDocProvider));

  // Window-scoped only. Without this the collection persists across
  // VS Code restarts, which means terminals opened during the next
  // session's activation gap (or restored from the prior session)
  // capture STALE port/token values from the prior server. New
  // terminals opened post-activation got the right values, but
  // pre-activation terminals silently held a port the new server
  // wasn't listening on. Setting persistent = false clears the
  // collection between sessions; pre-activation terminals see the
  // env vars unset (clean failure) rather than wrong (silent failure).
  context.environmentVariableCollection.persistent = false;

  startServer({
    onReady: (info) => {
      const env = context.environmentVariableCollection;
      env.replace('COMPOSER_WS_PORT', String(info.port));
      env.replace('COMPOSER_AUTH_TOKEN', info.token);
      diagnostics?.serverReady(info.port);
      vscode.window.showInformationMessage(`Composer bridge listening on port ${info.port}`);
    },
    onServerError: (err) => diagnostics?.serverError(err),
    onConnectionOpened: (remote) => diagnostics?.connectionOpened(remote),
    onConnectionRejected: (remote, reason) => diagnostics?.connectionRejected(remote, reason),
    onConnectionClosed: (code, reason) => diagnostics?.connectionClosed(code, reason),
    onConnectionError: (err) => diagnostics?.connectionError(err),
  });
}

export async function deactivate(): Promise<void> {
  await stopServer();
  diagnostics?.serverStopped();
  diagnostics = null;
}
