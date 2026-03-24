import * as vscode from 'vscode';
import * as path from 'path';
import { registerMethod } from '../protocol';
import { VirtualDocProvider } from '../providers/virtualDoc';

const LANGUAGE_MAP: Record<string, string> = {
  '.ts': 'typescript', '.js': 'javascript', '.py': 'python', '.sol': 'solidity',
  '.json': 'json', '.md': 'markdown', '.yaml': 'yaml', '.yml': 'yaml',
  '.toml': 'toml', '.rs': 'rust', '.go': 'go', '.spec': 'cvl',
  '.txt': 'plaintext', '.html': 'html', '.css': 'css',
};

let showFileCounter = 0;

function workspaceRoot(): vscode.Uri {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) {
    throw new Error('No workspace folder open');
  }
  return folders[0].uri;
}

export function registerWorkspaceHandlers(virtualDoc: VirtualDocProvider): void {
  registerMethod('workspace/getRoot', async () => {
    return { path: workspaceRoot().fsPath };
  });

  registerMethod('workspace/readFile', async (params) => {
    const filePath = params.path as string;
    if (!filePath) {
      throw new Error('Missing required param: path');
    }

    const uri = vscode.Uri.joinPath(workspaceRoot(), filePath);
    const data = await vscode.workspace.fs.readFile(uri);
    const content = Buffer.from(data).toString('utf-8');
    const ext = path.extname(filePath).toLowerCase();

    return { content, language: LANGUAGE_MAP[ext] ?? 'plaintext' };
  });

  registerMethod('workspace/listFiles', async (params) => {
    const dirPath = (params.path as string | undefined) ?? '';
    const pattern = params.pattern as string | undefined;

    if (pattern) {
      const uris = await vscode.workspace.findFiles(pattern);
      return {
        files: uris.map((uri) => ({
          name: path.basename(uri.fsPath),
          type: 'file' as const,
          path: vscode.workspace.asRelativePath(uri),
        })),
      };
    }

    const uri = vscode.Uri.joinPath(workspaceRoot(), dirPath);
    const entries = await vscode.workspace.fs.readDirectory(uri);
    return {
      files: entries.map(([name, type]) => ({
        name,
        type: type === vscode.FileType.Directory ? 'directory' as const : 'file' as const,
        path: dirPath ? `${dirPath}/${name}` : name,
      })),
    };
  });

  registerMethod('workspace/showFile', async (params) => {
    const content = params.content as string;
    const filePath = params.path as string;

    if (content === undefined || !filePath) {
      throw new Error('Missing required params: content, path');
    }

    const uri = virtualDoc.set(`/show/${showFileCounter++}/${filePath}`, content);
    const doc = await vscode.workspace.openTextDocument(uri);

    const lang = params.lang as string | undefined;
    if (lang) {
      await vscode.languages.setTextDocumentLanguage(doc, lang);
    }

    await vscode.window.showTextDocument(doc, { preview: true });

    return { success: true };
  });
}
