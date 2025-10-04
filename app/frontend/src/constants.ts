/*
Constants for App Configs... maybe I name this config
*/
export const WS_CONFIG = {
  URL: `ws://localhost:8080/ws/generate/`,
  RECONNECT_DELAY: 2000, // in ms
  MAX_RECONNECT_ATTEMPTS: 5,
} as const;

export const CANVAS_CONFIG = {
  BACKGROUND_COLOR: "#000000",
  JPEG_QUALITY: 90,
} as const;

export const PERF_CONFIG = {
  FPS_UPDATE_INTERVAL: 1000, // update interval in ms AKA: how often to recalculate FPS
  LATENCY_SAMPLE_SIZE: 10, // number of latency samples to average : Future use averaging ping times
} as const;
