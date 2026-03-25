export interface JsonRpcRequest {
  jsonrpc: '2.0';
  id: number | string;
  method: string;
  params?: Record<string, unknown>;
}

export interface JsonRpcResponse {
  jsonrpc: '2.0';
  id: number | string;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
}

type Handler = (params: Record<string, unknown>) => Promise<unknown>;

const handlers = new Map<string, Handler>();

export function registerMethod(method: string, handler: Handler): void {
  handlers.set(method, handler);
}

export async function dispatch(raw: string): Promise<string | null> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return JSON.stringify({
      jsonrpc: '2.0',
      id: null,
      error: { code: -32700, message: 'Parse error' },
    });
  }

  const req = parsed as JsonRpcRequest;
  if (req.jsonrpc !== '2.0' || req.method === undefined || req.id === undefined) {
    return JSON.stringify({
      jsonrpc: '2.0',
      id: req.id ?? null,
      error: { code: -32600, message: 'Invalid request' },
    });
  }

  const handler = handlers.get(req.method);
  if (!handler) {
    return JSON.stringify({
      jsonrpc: '2.0',
      id: req.id,
      error: { code: -32601, message: `Method not found: ${req.method}` },
    });
  }

  try {
    const result = await handler(req.params ?? {});
    return JSON.stringify({ jsonrpc: '2.0', id: req.id, result });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return JSON.stringify({
      jsonrpc: '2.0',
      id: req.id,
      error: { code: -32000, message },
    });
  }
}
