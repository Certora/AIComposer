import * as crypto from 'crypto';
import { WebSocketServer, WebSocket } from 'ws';
import { IncomingMessage } from 'http';
import { dispatch } from './protocol';

export interface ServerInfo {
  port: number;
  token: string;
}

/**
 * Lifecycle hooks the extension uses to drive its diagnostics surface.
 *
 * Every callback is optional so unit-testing the server doesn't have to
 * stub the whole observability stack — but the real activate() wires all
 * of them so that "did the extension crash?" becomes answerable.
 */
export interface ServerListeners {
  onReady?: (info: ServerInfo) => void;
  onServerError?: (err: unknown) => void;
  onConnectionOpened?: (remote: string) => void;
  onConnectionRejected?: (remote: string, reason: string) => void;
  onConnectionClosed?: (code: number, reason: string) => void;
  onConnectionError?: (err: unknown) => void;
}

let wss: WebSocketServer | null = null;

export function startServer(listeners: ServerListeners): void {
  const token = crypto.randomBytes(32).toString('hex');

  wss = new WebSocketServer({
    host: '127.0.0.1',
    port: 0,
    // Cap inbound frames so a runaway client can't OOM the host.
    maxPayload: 16 * 1024 * 1024,
  });

  // CRITICAL: previously absent. An unhandled 'error' event on an
  // EventEmitter becomes an uncaught exception that crashes the
  // extension host. Surface it to diagnostics instead.
  wss.on('error', (err) => {
    listeners.onServerError?.(err);
  });

  wss.on('listening', () => {
    const addr = wss!.address();
    if (typeof addr === 'object' && addr !== null) {
      listeners.onReady?.({ port: addr.port, token });
    }
  });

  wss.on('connection', (ws: WebSocket, req: IncomingMessage) => {
    const remote = `${req.socket.remoteAddress ?? '?'}:${req.socket.remotePort ?? '?'}`;
    const url = new URL(req.url ?? '', `http://${req.headers.host}`);
    if (url.searchParams.get('token') !== token) {
      listeners.onConnectionRejected?.(remote, 'bad or missing token');
      ws.close(4401, 'Unauthorized');
      return;
    }

    listeners.onConnectionOpened?.(remote);

    // Same rationale as wss.on('error'): per-connection 'error' events
    // crash the host if unhandled (e.g. ECONNRESET mid-message).
    ws.on('error', (err) => {
      listeners.onConnectionError?.(err);
    });

    ws.on('close', (code, reason) => {
      listeners.onConnectionClosed?.(code, reason.toString('utf-8'));
    });

    ws.on('message', async (data: Buffer) => {
      try {
        const response = await dispatch(data.toString());
        if (response) {
          ws.send(response);
        }
      } catch (err) {
        // dispatch() already wraps handler errors as JSON-RPC error
        // responses, so reaching this catch implies an internal bug
        // (e.g. JSON.stringify of a circular value). Surface and keep
        // the connection alive.
        listeners.onConnectionError?.(err);
      }
    });
  });
}

export async function stopServer(): Promise<void> {
  if (!wss) {
    return;
  }
  const server = wss;
  wss = null;
  for (const client of server.clients) {
    client.close();
  }
  await new Promise<void>((resolve) => server.close(() => resolve()));
}
