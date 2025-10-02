/**
 * Type definitions for the StreamDiffusion visualizer
 */

/**
 * WebSocket connection status
 */
export const ConnectionStatus = {
  DISCONNECTED: 'disconnected',
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  ERROR: 'error',
} as const;

export type ConnectionStatus = typeof ConnectionStatus[keyof typeof ConnectionStatus];

/**
 * Performance metrics from the backend
 * These will be populated when StreamDiffusion is integrated
 */
export interface BackendMetrics {
  /** Inference time in milliseconds (instantaneous) */
  inferenceTimeMs?: number;
  /** Exponential moving average of inference time in milliseconds */
  inferenceTimeEmaMs?: number;
  /** Frame dimensions */
  frameWidth?: number;
  frameHeight?: number;
}

/**
 * Client-side performance metrics
 */
export interface ClientMetrics {
  /** Frames received per second */
  fps: number;
  /** WebSocket latency in milliseconds */
  latencyMs?: number;
}

/**
 * Combined metrics for display
 */
export interface Metrics extends ClientMetrics, BackendMetrics {}

/**
 * WebSocket message types
 */
export type WebSocketMessage =
  | { type: 'metrics'; data: BackendMetrics }
  | { type: 'frame'; data: ArrayBuffer };
