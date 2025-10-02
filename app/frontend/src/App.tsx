import { useRef, useCallback } from 'react';
import { Canvas } from './components/Canvas';
import type { CanvasHandle } from './components/Canvas';
import { Controls } from './components/Controls';
import { ErrorBoundary } from './components/ErrorBoundary';
import { useWebSocket } from './hooks/useWebSocket';
import { WS_CONFIG } from './constants';
import type { Metrics } from './types';

/**
 * Main application component
 *
 * ARCHITECTURE OVERVIEW:
 * - Canvas: High-performance rendering (imperative, outside React)
 * - Controls: UI overlay with metrics and connection controls
 * - useWebSocket: Manages WebSocket connection and frame streaming
 * - ErrorBoundary: Catches and handles errors gracefully
 *
 * DATA FLOW:
 * 1. User clicks "Connect" in Controls
 * 2. useWebSocket opens WebSocket connection
 * 3. Backend sends JPEG frames as binary data
 * 4. onFrame callback receives ArrayBuffer
 * 5. Canvas.renderFrame() decodes and draws to canvas
 * 6. FPS is calculated and displayed in Controls
 *
 * PERFORMANCE NOTES:
 * - Frame rendering bypasses React's render cycle (imperative API)
 * - useCallback ensures stable function references
 * - useRef pattern in useWebSocket prevents reconnections
 *
 * LEARNING REACT:
 * - Composition: Small, focused components working together
 * - Props drilling: Parent controls child behavior via props
 * - Refs: Access child imperative methods (Canvas.renderFrame)
 * - Hooks: Encapsulate stateful logic (useWebSocket)
 */
function App() {
  /**
   * Reference to Canvas component for imperative rendering
   */
  const canvasRef = useRef<CanvasHandle>(null);

  /**
   * Frame handler: called when WebSocket receives a frame
   * Wrapped in useCallback to maintain stable reference
   * (though useWebSocket uses useRef pattern, this is still good practice)
   */
  const handleFrame = useCallback(async (data: ArrayBuffer) => {
    await canvasRef.current?.renderFrame(data);
  }, []);

  /**
   * WebSocket hook for connection management and frame streaming
   */
  const { connect, disconnect, status, fps, reconnectAttempts } = useWebSocket({
    url: WS_CONFIG.URL,
    onFrame: handleFrame,
    autoConnect: false,
    enableReconnect: true,
  });

  /**
   * Combine metrics for display
   */
  const metrics: Metrics = {
    fps,
    // Backend metrics will be added here when StreamDiffusion is integrated
    // inferenceTimeEmaMs: ...,
    // latencyMs: ...,
  };

  return (
    <ErrorBoundary>
      {/* Full-screen container */}
      <div className="relative w-screen h-screen bg-black overflow-hidden">

        {/* Canvas: fills entire screen */}
        <Canvas ref={canvasRef} className="w-full h-full" />

        {/* Controls: overlay panel */}
        <Controls
          status={status}
          metrics={metrics}
          onConnect={connect}
          onDisconnect={disconnect}
          reconnectAttempts={reconnectAttempts}
        />

      </div>
    </ErrorBoundary>
  );
}

export default App;
