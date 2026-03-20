import * as vscode from 'vscode';

export class VirtualDocProvider implements vscode.TextDocumentContentProvider {
  private contents = new Map<string, string>();
  private _onDidChange = new vscode.EventEmitter<vscode.Uri>();
  readonly onDidChange = this._onDidChange.event;

  set(uriPath: string, content: string): vscode.Uri {
    this.contents.set(uriPath, content);
    const uri = vscode.Uri.parse(`composer:${uriPath}`);
    this._onDidChange.fire(uri);
    return uri;
  }

  delete(uriPath: string): void {
    this.contents.delete(uriPath);
  }

  provideTextDocumentContent(uri: vscode.Uri): string {
    return this.contents.get(uri.path) ?? '';
  }

  dispose(): void {
    this._onDidChange.dispose();
  }
}
