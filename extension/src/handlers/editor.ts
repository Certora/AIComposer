import * as vscode from 'vscode';
import { registerMethod } from '../protocol';
import { VirtualDocProvider } from '../providers/virtualDoc';

let counter = 0;

export function registerEditorHandlers(provider: VirtualDocProvider): void {
  registerMethod('editor/showDiff', async (params) => {
    const originalContent = params.originalContent as string;
    const modifiedContent = params.modifiedContent as string;
    const title = (params.title as string | undefined) ?? 'Diff';

    if (originalContent === undefined || modifiedContent === undefined) {
      throw new Error('Missing required params: originalContent, modifiedContent');
    }

    const id = counter++;
    const originalUri = provider.set(`/diff/${id}/original`, originalContent);
    const modifiedUri = provider.set(`/diff/${id}/modified`, modifiedContent);

    await vscode.commands.executeCommand('vscode.diff', originalUri, modifiedUri, title);

    return { success: true };
  });
}
