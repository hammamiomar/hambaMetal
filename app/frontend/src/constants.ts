/**
 * Application constants and configuration
 */

/**
 * WebSocket configuration
 */
export const WS_CONFIG = {
  /** WebSocket URL for streaming */
  URL: `ws://localhost:8080/ws/test/`,

  /** Reconnection attempt delay in milliseconds */
  RECONNECT_DELAY: 2000,

  /** Maximum reconnection attempts before giving up */
  MAX_RECONNECT_ATTEMPTS: 5,
} as const;

/**
 * Canvas configuration
 */
export const CANVAS_CONFIG = {
  /** Default background color when no frame is displayed */
  BACKGROUND_COLOR: '#000000',

  /** JPEG quality for encoding (if needed) */
  JPEG_QUALITY: 90,
} as const;

/**
 * Performance monitoring
 */
export const PERF_CONFIG = {
  /** Interval for FPS calculation in milliseconds */
  FPS_UPDATE_INTERVAL: 1000,

  /** Number of latency samples to average */
  LATENCY_SAMPLE_SIZE: 10,
} as const;
