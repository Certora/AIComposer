import * as vscode from 'vscode';
import { ChatViewProvider } from './chatViewProvider';
import { ContextManagerProvider, ContextItem } from './contextManagerProvider';
import { DashboardProvider } from './dashboardProvider';
import { SettingsProvider } from './settingsProvider';

export function activate(context: vscode.ExtensionContext) {
    console.log('Certora AI Composer extension is now active');

    const contextManagerProvider = new ContextManagerProvider();
    vscode.window.registerTreeDataProvider(
        'certora-ai-composer.contextManager',
        contextManagerProvider
    );

    // Pre-load default trivial example files
    const loadTrivial = async () => {
        const categories: { [key: string]: string } = {
            'Intf.sol': 'Interfaces',
            'simple.spec': 'Specs',
            'system_doc_simple.txt': 'System Docs',
            'README.md': 'System Docs'
        };

        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (!workspaceFolders) return;

        for (const [fileName, cat] of Object.entries(categories)) {
            // Strategy 1: Direct path relative to workspace root
            const directPath = vscode.Uri.joinPath(workspaceFolders[0].uri, 'examples', 'trivial', fileName);
            try {
                await vscode.workspace.fs.stat(directPath);
                contextManagerProvider.addFile(cat, directPath.fsPath);
                continue; // Found it
            } catch (e) {}

            // Strategy 2: findFiles as fallback
            const found = await vscode.workspace.findFiles(`**/examples/trivial/${fileName}`, '**/node_modules/**', 1);
            if (found.length > 0) {
                contextManagerProvider.addFile(cat, found[0].fsPath);
            }
        }
    };
    
    // Give the tree view a moment to initialize before adding files
    setTimeout(loadTrivial, 1000);

    const chatViewProvider = new ChatViewProvider(context.extensionUri, contextManagerProvider);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            'certora-ai-composer.chatView',
            chatViewProvider
        )
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('certora-ai-composer.removeContext', (item: ContextItem) => {
            contextManagerProvider.removeFile(item);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('certora-ai-composer.openFile', (filePath: string) => {
            vscode.window.showTextDocument(vscode.Uri.file(filePath));
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('certora-ai-composer.openDashboard', () => {
            DashboardProvider.createOrShow(context.extensionUri);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('certora-ai-composer.openSettings', () => {
            SettingsProvider.createOrShow(context.extensionUri);
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('certora-ai-composer.addContext', async () => {
            const categories = ['Interfaces', 'Specs', 'System Docs'];
            const category = await vscode.window.showQuickPick(categories, {
                placeHolder: 'Select a category to add to'
            });

            if (category) {
                const files = await vscode.window.showOpenDialog({
                    canSelectMany: true,
                    openLabel: 'Add to context'
                });

                if (files) {
                    files.forEach((file: vscode.Uri) => {
                        contextManagerProvider.addFile(category, file.fsPath);
                    });
                }
            }
        })
    );
}

export function deactivate() {}

