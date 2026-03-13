import * as vscode from 'vscode';

interface FileEntry {
  relativePath: string;
  content: string;
  existsInWorkspace: boolean;
}

interface TreeNode {
  label: string;
  fullPath: string;
  children: Map<string, TreeNode>;
  file?: FileEntry;
}

export class ProposedTreeProvider implements vscode.TreeDataProvider<string> {
  private _onDidChangeTreeData = new vscode.EventEmitter<string | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private files = new Map<string, FileEntry>();
  private root: TreeNode = { label: '', fullPath: '', children: new Map() };

  populate(fileMap: Record<string, string>, existingPaths: Set<string>): void {
    this.files.clear();
    this.root = { label: '', fullPath: '', children: new Map() };

    for (const [path, content] of Object.entries(fileMap)) {
      const entry: FileEntry = {
        relativePath: path,
        content,
        existsInWorkspace: existingPaths.has(path),
      };
      this.files.set(path, entry);

      // Build tree structure
      const parts = path.split('/');
      let node = this.root;
      for (let i = 0; i < parts.length; i++) {
        const part = parts[i];
        if (!node.children.has(part)) {
          const fullPath = parts.slice(0, i + 1).join('/');
          node.children.set(part, { label: part, fullPath, children: new Map() });
        }
        node = node.children.get(part)!;
      }
      node.file = entry;
    }

    vscode.commands.executeCommand('setContext', 'composer.hasProposedChanges', true);
    this._onDidChangeTreeData.fire(undefined);
    vscode.commands.executeCommand('composerProposedChanges.focus');
  }

  clear(): void {
    this.files.clear();
    this.root = { label: '', fullPath: '', children: new Map() };
    vscode.commands.executeCommand('setContext', 'composer.hasProposedChanges', false);
    this._onDidChangeTreeData.fire(undefined);
  }

  getFile(path: string): FileEntry | undefined {
    return this.files.get(path);
  }

  allFiles(): Map<string, string> {
    const result = new Map<string, string>();
    for (const [path, entry] of this.files) {
      result.set(path, entry.content);
    }
    return result;
  }

  getTreeItem(element: string): vscode.TreeItem {
    const node = this.findNode(element);
    if (!node) {
      return new vscode.TreeItem(element);
    }

    const isFile = node.file !== undefined;
    const item = new vscode.TreeItem(
      node.label,
      isFile ? vscode.TreeItemCollapsibleState.None : vscode.TreeItemCollapsibleState.Expanded,
    );

    if (isFile) {
      item.contextValue = 'proposedFile';
      item.tooltip = node.fullPath;

      if (node.file!.existsInWorkspace) {
        item.description = 'M';
        item.iconPath = new vscode.ThemeIcon('diff-modified', new vscode.ThemeColor('gitDecoration.modifiedResourceForeground'));
      } else {
        item.description = '+';
        item.iconPath = new vscode.ThemeIcon('diff-added', new vscode.ThemeColor('gitDecoration.addedResourceForeground'));
      }

      item.command = {
        command: 'composer.openProposedFile',
        title: 'Open Proposed File',
        arguments: [node.fullPath],
      };
    } else {
      item.iconPath = vscode.ThemeIcon.Folder;
    }

    return item;
  }

  getChildren(element?: string): string[] {
    const node = element ? this.findNode(element) : this.root;
    if (!node) {
      return [];
    }
    return [...node.children.values()].map((n) => n.fullPath);
  }

  private findNode(path: string): TreeNode | undefined {
    const parts = path.split('/');
    let node = this.root;
    for (const part of parts) {
      const child = node.children.get(part);
      if (!child) {
        return undefined;
      }
      node = child;
    }
    return node;
  }

  dispose(): void {
    this._onDidChangeTreeData.dispose();
  }
}
