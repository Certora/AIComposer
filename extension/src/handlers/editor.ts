import * as vscode from 'vscode';
import { registerMethod } from '../protocol';
import { VirtualDocProvider } from '../providers/virtualDoc';

// VS Code's `vscode.diff` command returns void — there's no way to get a
// handle to the resulting tab. We embed a unique marker in the title and
// match against `tab.label` to find it again when closing.
const TITLE_MARKER_PREFIX = '[composer:';
const TITLE_MARKER_SUFFIX = ']';

function titleMarker(diffId: string): string {
  return `${TITLE_MARKER_PREFIX}${diffId}${TITLE_MARKER_SUFFIX}`;
}

let counter = 0;

export function registerEditorHandlers(provider: VirtualDocProvider): void {
  registerMethod('editor/showDiff', async (params) => {
    const originalContent = params.originalContent as string;
    const modifiedContent = params.modifiedContent as string;
    const baseTitle = (params.title as string | undefined) ?? 'Diff';

    if (originalContent === undefined || modifiedContent === undefined) {
      throw new Error('Missing required params: originalContent, modifiedContent');
    }

    const diffId = String(counter++);
    const originalUri = provider.set(`/diff/${diffId}/original`, originalContent);
    const modifiedUri = provider.set(`/diff/${diffId}/modified`, modifiedContent);
    const title = `${baseTitle} ${titleMarker(diffId)}`;

    await vscode.commands.executeCommand('vscode.diff', originalUri, modifiedUri, title);

    return { diffId };
  });

  registerMethod('editor/closeDiff', async (params) => {
    const diffId = params.diffId as string | undefined;
    if (diffId === undefined) {
      throw new Error('Missing required param: diffId');
    }

    const marker = titleMarker(diffId);
    const tabsToClose: vscode.Tab[] = [];
    for (const group of vscode.window.tabGroups.all) {
      for (const tab of group.tabs) {
        if (tab.input instanceof vscode.TabInputTextDiff && tab.label.endsWith(marker)) {
          tabsToClose.push(tab);
        }
      }
    }
    if (tabsToClose.length > 0) {
      await vscode.window.tabGroups.close(tabsToClose);
    }

    provider.delete(`/diff/${diffId}/original`);
    provider.delete(`/diff/${diffId}/modified`);

    return { closed: tabsToClose.length > 0 };
  });
}
