// WebSocket connection status
export const ConnectionStatus = {
  DISCONNECTED: "disconnected",
  CONNECTING: "connecting",
  CONNECTED: "connected",
  ERROR: "error",
} as const;

export type ConnectionStatus =
  (typeof ConnectionStatus)[keyof typeof ConnectionStatus];
// ConnectionStatus enforced strict type values : = 'disconnected' | 'connecting' | 'connected' | 'error'

export interface BackendMetrics {
  // Inference time in milliseconds (instantaneous)
  inferenceTimeMs?: number;
  // Exponential moving average of inference time in milliseconds
  inferenceTimeEmaMs?: number;
  frameWidth?: number;
  frameHeight?: number;
}

export interface ClientMetrics {
  fps: number;
  // WebSocket latency in milliseconds
  latencyMs?: number;
}
export interface Metrics extends ClientMetrics, BackendMetrics {}

// Discriminated union type... if a var is of type WebSocketMessage they can do:
//  ex: msg.type == 'metrics'--> compiler knows data type
export type WebSocketMessage =
  | { type: "metrics"; data: BackendMetrics }
  | { type: "frame"; data: ArrayBuffer };
