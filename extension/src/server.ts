import * as crypto from 'crypto';
import { WebSocketServer, WebSocket } from 'ws';
import { IncomingMessage } from 'http';
import { dispatch } from './protocol';

export interface ServerInfo {
  port: number;
  token: string;
}

let wss: WebSocketServer | null = null;

export function startServer(onReady: (info: ServerInfo) => void): void {
  const token = crypto.randomBytes(32).toString('hex');

  wss = new WebSocketServer({ host: '127.0.0.1', port: 0 }, () => {
    const addr = wss!.address()!;
    if (typeof addr === 'object') {
      onReady({ port: addr.port, token });
    }
  });

  wss.on('connection', (ws: WebSocket, req: IncomingMessage) => {
    const url = new URL(req.url ?? '', `http://${req.headers.host}`);
    if (url.searchParams.get('token') !== token) {
      ws.close(4401, 'Unauthorized');
      return;
    }

    ws.on('message', async (data: Buffer) => {
      const response = await dispatch(data.toString());
      if (response) {
        ws.send(response);
      }
    });
  });
}

export function stopServer(): void {
  if (wss) {
    for (const client of wss.clients) {
      client.close();
    }
    wss.close();
    wss = null;
  }
}
