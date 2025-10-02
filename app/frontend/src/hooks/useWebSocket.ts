import { useEffect, useRef, useCallback, useState } from 'react';
import { ConnectionStatus } from '../types';
import { WS_CONFIG, PERF_CONFIG } from '../constants';

/**
 * Configuration options for the WebSocket hook
 */
interface UseWebSocketOptions {
  /** WebSocket URL to connect to */
  url: string;

  /**
   * Callback fired when a frame is received
   * NOTE: This callback is stored in a ref to prevent reconnections
   * when the parent component re-renders. See implementation below.
   */
  onFrame?: (data: ArrayBuffer) => void;

  /** Auto-connect on mount */
  autoConnect?: boolean;

  /** Enable automatic reconnection on disconnect */
  enableReconnect?: boolean;
}

/**
 * Return type for the useWebSocket hook
 */
interface UseWebSocketReturn {
  /** Initiate WebSocket connection */
  connect: () => void;

  /** Close WebSocket connection */
  disconnect: () => void;

  /** Current connection status */
  status: ConnectionStatus;

  /** Frames received per second */
  fps: number;

  /** Number of reconnection attempts (for debugging) */
  reconnectAttempts: number;
}

/**
 * Custom hook for managing WebSocket connections with frame streaming
 *
 * ARCHITECTURE NOTES:
 * - Uses the "latest ref" pattern to avoid reconnection issues
 * - The onFrame callback is stored in a ref and updated on every render
 * - This prevents the connect() function from being recreated when onFrame changes
 * - Without this pattern, every parent re-render would disconnect/reconnect the WebSocket
 *
 * LEARNING REACT:
 * This is a critical pattern when building hooks that manage subscriptions.
 * Always ask: "Does this dependency ACTUALLY need to trigger cleanup/re-run?"
 * If the answer is "no, I just need the latest value", use a ref.
 *
 * @param options - Configuration options
 * @returns WebSocket connection controls and metrics
 */
export function useWebSocket({
  url,
  onFrame,
  autoConnect = false,
  enableReconnect = true,
}: UseWebSocketOptions): UseWebSocketReturn {
  // WebSocket instance (stored in ref to persist across renders)
  const ws = useRef<WebSocket | null>(null);

  // Connection state
  const [status, setStatus] = useState<ConnectionStatus>(ConnectionStatus.DISCONNECTED);

  // FPS tracking
  const [fps, setFps] = useState(0);
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(Date.now());

  // Reconnection tracking
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const reconnectTimeoutRef = useRef<number | null>(null);

  /**
   * THE CRITICAL PATTERN: Store the latest onFrame callback in a ref
   * This ref is updated on every render, but doesn't trigger effects
   */
  const onFrameRef = useRef(onFrame);

  // Update the ref whenever onFrame changes (no effect re-run!)
  useEffect(() => {
    onFrameRef.current = onFrame;
  });

  /**
   * Connect to WebSocket
   * NOTE: url is the only dependency - onFrame is accessed via ref
   */
  const connect = useCallback(() => {
    // Prevent duplicate connections
    if (ws.current?.readyState === WebSocket.OPEN || ws.current?.readyState === WebSocket.CONNECTING) {
      console.log('[useWebSocket] Already connected or connecting');
      return;
    }

    console.log(`[useWebSocket] Connecting to ${url}`);
    setStatus(ConnectionStatus.CONNECTING);

    ws.current = new WebSocket(url);
    ws.current.binaryType = 'arraybuffer';

    ws.current.onopen = () => {
      console.log('[useWebSocket] Connected');
      setStatus(ConnectionStatus.CONNECTED);
      setReconnectAttempts(0);
    };

    ws.current.onmessage = (event: MessageEvent) => {
      // Handle binary frame data
      if (event.data instanceof ArrayBuffer) {
        // Call the LATEST onFrame callback via ref
        if (onFrameRef.current) {
          onFrameRef.current(event.data);
        }

        // Update FPS counter
        frameCountRef.current++;
        const now = Date.now();
        if (now - lastFpsUpdateRef.current >= PERF_CONFIG.FPS_UPDATE_INTERVAL) {
          setFps(frameCountRef.current);
          frameCountRef.current = 0;
          lastFpsUpdateRef.current = now;
        }
      }
      // Handle JSON metadata (for future use)
      else if (typeof event.data === 'string') {
        try {
          const metadata = JSON.parse(event.data);
          console.log('[useWebSocket] Received metadata:', metadata);
          // TODO: Handle backend metrics here
        } catch (e) {
          console.warn('[useWebSocket] Failed to parse JSON message:', e);
        }
      }
    };

    ws.current.onerror = (error) => {
      console.error('[useWebSocket] Error:', error);
      setStatus(ConnectionStatus.ERROR);
    };

    ws.current.onclose = (event) => {
      console.log(`[useWebSocket] Disconnected (code: ${event.code}, reason: ${event.reason})`);
      ws.current = null;
      setStatus(ConnectionStatus.DISCONNECTED);
      setFps(0);

      // Attempt reconnection if enabled and not a normal closure
      if (enableReconnect && event.code !== 1000 && reconnectAttempts < WS_CONFIG.MAX_RECONNECT_ATTEMPTS) {
        const attempt = reconnectAttempts + 1;
        setReconnectAttempts(attempt);
        console.log(`[useWebSocket] Reconnecting in ${WS_CONFIG.RECONNECT_DELAY}ms (attempt ${attempt}/${WS_CONFIG.MAX_RECONNECT_ATTEMPTS})`);

        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect();
        }, WS_CONFIG.RECONNECT_DELAY);
      }
    };
  }, [url, enableReconnect, reconnectAttempts]); // onFrame is NOT in dependencies!

  /**
   * Disconnect from WebSocket
   */
  const disconnect = useCallback(() => {
    // Clear any pending reconnection attempts
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (ws.current) {
      console.log('[useWebSocket] Disconnecting');
      ws.current.close(1000, 'Client requested disconnect');
      ws.current = null;
      setStatus(ConnectionStatus.DISCONNECTED);
      setFps(0);
      setReconnectAttempts(0);
    }
  }, []);

  /**
   * Auto-connect effect (only runs when autoConnect or url changes)
   */
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    connect,
    disconnect,
    status,
    fps,
    reconnectAttempts,
  };
}
