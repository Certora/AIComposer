import * as vscode from 'vscode';

export class ContextManagerProvider implements vscode.TreeDataProvider<ContextItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<ContextItem | undefined | void> = new vscode.EventEmitter<ContextItem | undefined | void>();
    readonly onDidChangeTreeData: vscode.Event<ContextItem | undefined | void> = this._onDidChangeTreeData.event;

    private items: { [key: string]: string[] } = {
        'Interfaces': [],
        'Specs': [],
        'System Docs': []
    };

    getTreeItem(element: ContextItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: ContextItem): vscode.ProviderResult<ContextItem[]> {
        if (!element) {
            return [
                new ContextItem(`Interfaces (${this.items['Interfaces'].length})`, vscode.TreeItemCollapsibleState.Expanded, 'folder', 'Interfaces'),
                new ContextItem(`Specs (${this.items['Specs'].length})`, vscode.TreeItemCollapsibleState.Expanded, 'folder', 'Specs'),
                new ContextItem(`System Docs (${this.items['System Docs'].length})`, vscode.TreeItemCollapsibleState.Expanded, 'folder', 'System Docs')
            ];
        }

        if (element.contextValue === 'folder') {
            const category = element.category as string;
            return this.items[category].map(file => {
                const item = new ContextItem(file.split('/').pop() || file, vscode.TreeItemCollapsibleState.None, 'file');
                item.category = category;
                item.fullPath = file;
                item.tooltip = file;
                item.command = {
                    command: 'certora-ai-composer.openFile',
                    title: 'Open File',
                    arguments: [file]
                };
                return item;
            });
        }

        return [];
    }

    addFile(category: string, filePath: string) {
        if (this.items[category]) {
            if (!this.items[category].includes(filePath)) {
                this.items[category].push(filePath);
                this._onDidChangeTreeData.fire();
            }
        }
    }

    removeFile(item: ContextItem) {
        if (item.category && this.items[item.category]) {
            this.items[item.category] = this.items[item.category].filter(f => f !== item.fullPath);
            this._onDidChangeTreeData.fire();
        }
    }

    getContext() {
        return {
            interfaces: this.items['Interfaces'],
            specs: this.items['Specs'],
            docs: this.items['System Docs']
        };
    }
}

export class ContextItem extends vscode.TreeItem {
    public fullPath?: string;

    constructor(
        public readonly label: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly contextValue: 'folder' | 'file',
        public category?: string
    ) {
        super(label, collapsibleState);
        this.tooltip = this.label;
        if (contextValue === 'file') {
            this.iconPath = new vscode.ThemeIcon('file');
            this.command = {
                command: 'certora-ai-composer.openFile',
                title: 'Open File',
                arguments: [] // Will be set after construction
            };
        } else {
            this.iconPath = new vscode.ThemeIcon('folder');
        }
    }
}

