import * as vscode from 'vscode';
import { registerMethod } from '../protocol';
import { ProposedTreeProvider } from '../providers/proposedTree';
import { VirtualDocProvider } from '../providers/virtualDoc';

interface Preview {
  id: string;
  files: Record<string, string>;
}

const previews = new Map<string, Preview>();
let previewCounter = 0;

async function computeExistingPaths(files: Record<string, string>): Promise<Set<string>> {
  const existing = new Set<string>();
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) {
    return existing;
  }
  const root = folders[0].uri;
  for (const relativePath of Object.keys(files)) {
    try {
      await vscode.workspace.fs.stat(vscode.Uri.joinPath(root, relativePath));
      existing.add(relativePath);
    } catch {
      // File doesn't exist in workspace
    }
  }
  return existing;
}

export function registerResultsHandlers(
  tree: ProposedTreeProvider,
  virtualDoc: VirtualDocProvider,
): void {
  registerMethod('results/preview', async (params) => {
    const files = params.files as Record<string, string>;
    if (!files || Object.keys(files).length === 0) {
      throw new Error('Missing required param: files');
    }

    const id = `preview-${previewCounter++}`;
    previews.set(id, { id, files });

    const existing = await computeExistingPaths(files);
    tree.populate(files, existing);

    return { success: true, previewId: id };
  });

  registerMethod('results/accept', async (params) => {
    const previewId = params.previewId as string;
    const preview = previews.get(previewId);
    if (!preview) {
      throw new Error(`Unknown preview: ${previewId}`);
    }

    const folders = vscode.workspace.workspaceFolders;
    if (!folders || folders.length === 0) {
      throw new Error('No workspace folder open');
    }
    const root = folders[0].uri;

    const writtenFiles: string[] = [];
    for (const [relativePath, content] of Object.entries(preview.files)) {
      const uri = vscode.Uri.joinPath(root, relativePath);
      await vscode.workspace.fs.writeFile(uri, Buffer.from(content, 'utf-8'));
      writtenFiles.push(relativePath);
    }

    tree.clear();
    previews.delete(previewId);

    return { success: true, writtenFiles };
  });

  registerMethod('results/reject', async (params) => {
    const previewId = params.previewId as string;
    if (!previews.has(previewId)) {
      throw new Error(`Unknown preview: ${previewId}`);
    }

    tree.clear();
    previews.delete(previewId);

    return { success: true };
  });
}

export function registerResultsCommands(
  tree: ProposedTreeProvider,
  virtualDoc: VirtualDocProvider,
): vscode.Disposable[] {
  const openFile = vscode.commands.registerCommand('composer.openProposedFile', async (filePath: string) => {
    const file = tree.getFile(filePath);
    if (!file) {
      return;
    }

    const folders = vscode.workspace.workspaceFolders;
    if (file.existsInWorkspace && folders && folders.length > 0) {
      const originalUri = vscode.Uri.joinPath(folders[0].uri, filePath);
      const proposedUri = virtualDoc.set(`/proposed/${filePath}`, file.content);
      await vscode.commands.executeCommand('vscode.diff', originalUri, proposedUri, `${filePath} (Proposed)`);
    } else {
      const uri = virtualDoc.set(`/proposed/${filePath}`, file.content);
      await vscode.commands.executeCommand('vscode.open', uri);
    }
  });

  return [openFile];
}
