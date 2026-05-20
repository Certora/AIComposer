import * as vscode from 'vscode';

/**
 * Centralizes all observability for the bridge: an OutputChannel ("Composer
 * Bridge") that captures every interesting lifecycle event, and a status-bar
 * item that gives a one-glance answer to "is the server alive and how many
 * clients are connected?".
 *
 * Callers should funnel every server / connection event through this class
 * so that there is exactly one source of truth for "what happened".
 */
export class Diagnostics implements vscode.Disposable {
  private readonly channel: vscode.OutputChannel;
  private readonly statusItem: vscode.StatusBarItem;
  private port: number | null = null;
  private connections = 0;
  private lastError: string | null = null;

  constructor() {
    this.channel = vscode.window.createOutputChannel('Composer Bridge');
    this.statusItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100,
    );
    this.statusItem.command = 'composer.bridge.status';
    this.refreshStatus();
    this.statusItem.show();
  }

  // ---- server lifecycle ----------------------------------------------------

  serverReady(port: number): void {
    this.port = port;
    this.lastError = null;
    this.log(`server listening on 127.0.0.1:${port}`);
    this.refreshStatus();
  }

  serverError(err: unknown): void {
    const msg = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    this.lastError = msg;
    this.log(`SERVER ERROR — ${msg}`);
    if (err instanceof Error && err.stack) {
      this.channel.appendLine(err.stack);
    }
    this.refreshStatus();
  }

  serverStopped(): void {
    this.log(`server stopped`);
    this.port = null;
    this.connections = 0;
    this.refreshStatus();
  }

  // ---- per-connection events -----------------------------------------------

  connectionOpened(remote: string): void {
    this.connections += 1;
    this.log(`connection opened from ${remote} (active=${this.connections})`);
    this.refreshStatus();
  }

  connectionRejected(remote: string, reason: string): void {
    this.log(`connection REJECTED from ${remote}: ${reason}`);
  }

  connectionClosed(code: number, reason: string): void {
    this.connections = Math.max(0, this.connections - 1);
    this.log(`connection closed (code=${code} reason=${reason || '<none>'} active=${this.connections})`);
    this.refreshStatus();
  }

  connectionError(err: unknown): void {
    const msg = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    this.log(`connection error — ${msg}`);
  }

  // ---- per-request events --------------------------------------------------

  dispatchError(method: string, err: unknown): void {
    const msg = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    this.log(`dispatch error in ${method} — ${msg}`);
  }

  // ---- on-demand -----------------------------------------------------------

  /** Print a multi-line status snapshot to the channel and reveal it. */
  showStatus(): void {
    this.channel.appendLine('');
    this.channel.appendLine('--- composer bridge status ---');
    this.channel.appendLine(`  listening: ${this.port !== null ? `yes (port ${this.port})` : 'no'}`);
    this.channel.appendLine(`  active connections: ${this.connections}`);
    this.channel.appendLine(`  last error: ${this.lastError ?? '<none>'}`);
    this.channel.appendLine('------------------------------');
    this.channel.show(true);
  }

  // ---- internals -----------------------------------------------------------

  private log(message: string): void {
    const ts = new Date().toISOString();
    this.channel.appendLine(`[${ts}] ${message}`);
  }

  private refreshStatus(): void {
    if (this.port === null) {
      this.statusItem.text = '$(circle-slash) composer: down';
      this.statusItem.tooltip = this.lastError
        ? `Composer bridge is not listening. Last error: ${this.lastError}`
        : 'Composer bridge is not listening.';
      this.statusItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
      return;
    }
    this.statusItem.text = `$(broadcast) composer:${this.port} (${this.connections})`;
    this.statusItem.tooltip =
      `Composer bridge listening on port ${this.port}.\n` +
      `Active connections: ${this.connections}.\n` +
      `Click for details.`;
    this.statusItem.backgroundColor = undefined;
  }

  dispose(): void {
    this.statusItem.dispose();
    this.channel.dispose();
  }
}
