import { Fragment } from "react";
import {
  Menu,
  MenuButton,
  MenuItem,
  MenuItems,
  Transition,
} from "@headlessui/react";
import { ConnectionStatus } from "../types";
import type { Metrics } from "../types";

/**
 * Props for the Controls component
 */
interface ControlsProps {
  /** Current connection status */
  status: ConnectionStatus;

  /** Performance metrics to display */
  metrics: Metrics;

  /** Callback to initiate connection */
  onConnect: () => void;

  /** Callback to disconnect */
  onDisconnect: () => void;

  /** Number of reconnection attempts (for debugging) */
  reconnectAttempts?: number;
}

/**
 * Control panel component with connection controls and metrics display
 *
 * DESIGN NOTES:
 * - Glass-morphism aesthetic (blur, transparency, subtle borders)
 * - Fixed position overlay (doesn't interfere with canvas)
 * - Headless UI for accessible, unstyled components
 *
 * LEARNING REACT:
 * - This is a "presentational" component (UI only, no logic)
 * - All state and handlers are passed as props
 * - Makes it easy to test and reuse
 * - Parent controls the behavior, this just renders
 *
 * HEADLESS UI:
 * - Provides behavior (keyboard nav, focus management, ARIA)
 * - You provide styling (complete design freedom)
 * - Menu component: accessible dropdown with transitions
 *
 * @param props - Component props
 */
export function Controls({
  status,
  metrics,
  onConnect,
  onDisconnect,
  reconnectAttempts = 0,
}: ControlsProps) {
  const isConnected = status === ConnectionStatus.CONNECTED;
  const isConnecting = status === ConnectionStatus.CONNECTING;

  /**
   * Get status indicator color based on connection state
   */
  const getStatusColor = () => {
    switch (status) {
      case ConnectionStatus.CONNECTED:
        return "bg-green-500";
      case ConnectionStatus.CONNECTING:
        return "bg-yellow-500";
      case ConnectionStatus.ERROR:
        return "bg-red-500";
      default:
        return "bg-gray-500";
    }
  };

  /**
   * Get human-readable status text
   */
  const getStatusText = () => {
    switch (status) {
      case ConnectionStatus.CONNECTED:
        return "Connected";
      case ConnectionStatus.CONNECTING:
        return "Connecting...";
      case ConnectionStatus.ERROR:
        return "Error";
      default:
        return "Disconnected";
    }
  };

  return (
    <div className="absolute top-6 left-6 w-80 select-none">
      {/* Main panel with glass-morphism effect */}
      <div className="bg-black/80 backdrop-blur-lg rounded-2xl p-6 border border-white/10 shadow-2xl">
        {/* Header: Status indicator */}
        <div className="flex items-center gap-3 mb-6">
          <div className="relative">
            {/* Status dot */}
            <div
              className={`w-3 h-3 rounded-full ${getStatusColor()} ${isConnected ? "animate-pulse" : ""}`}
            />

            {/* Pulse ring for connected state */}
            {isConnected && (
              <div className="absolute inset-0 w-3 h-3 rounded-full bg-green-500 animate-ping opacity-75" />
            )}
          </div>

          <h2 className="text-lg font-semibold text-white">
            {getStatusText()}
          </h2>

          {reconnectAttempts > 0 && (
            <span className="ml-auto text-xs text-gray-400">
              Retry {reconnectAttempts}
            </span>
          )}
        </div>

        {/* Connection controls */}
        <div className="space-y-3 mb-6">
          <button
            onClick={onConnect}
            disabled={isConnected || isConnecting}
            className="w-full px-4 py-2.5 bg-green-600 hover:bg-green-700 active:bg-green-800
                     disabled:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-50
                     text-white rounded-lg font-medium transition-all duration-150
                     focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 focus:ring-offset-black/80"
          >
            {isConnecting ? "Connecting..." : "Connect"}
          </button>

          <button
            onClick={onDisconnect}
            disabled={!isConnected && !isConnecting}
            className="w-full px-4 py-2.5 bg-red-600 hover:bg-red-700 active:bg-red-800
                     disabled:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-50
                     text-white rounded-lg font-medium transition-all duration-150
                     focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 focus:ring-offset-black/80"
          >
            Disconnect
          </button>
        </div>

        {/* Metrics display */}
        <div className="pt-6 border-t border-white/10 space-y-3">
          <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wide">
            Performance
          </h3>

          {/* Client FPS */}
          <MetricRow
            label="Client FPS"
            value={metrics.fps}
            unit=""
            highlight={isConnected}
          />

          {/* Backend metrics (placeholder for future) */}
          {metrics.inferenceTimeEmaMs !== undefined && (
            <MetricRow
              label="Inference Time"
              value={metrics.inferenceTimeEmaMs.toFixed(1)}
              unit="ms"
              highlight={isConnected}
            />
          )}

          {/* Latency (placeholder for future) */}
          {metrics.latencyMs !== undefined && (
            <MetricRow
              label="Latency"
              value={metrics.latencyMs.toFixed(1)}
              unit="ms"
              highlight={isConnected}
            />
          )}
        </div>

        {/* Settings menu (using Headless UI) */}
        <div className="pt-6 border-t border-white/10 mt-6">
          <Menu as="div" className="relative">
            {/* Menu button */}
            <MenuButton
              className="w-full px-4 py-2.5 bg-gray-800 hover:bg-gray-700 active:bg-gray-600
                                   text-white rounded-lg font-medium transition-all duration-150
                                   focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 focus:ring-offset-black/80"
            >
              Settings
            </MenuButton>

            {/* Menu items with transition */}
            <Transition
              as={Fragment}
              enter="transition ease-out duration-100"
              enterFrom="transform opacity-0 scale-95"
              enterTo="transform opacity-100 scale-100"
              leave="transition ease-in duration-75"
              leaveFrom="transform opacity-100 scale-100"
              leaveTo="transform opacity-0 scale-95"
            >
              <MenuItems
                className="absolute left-0 bottom-full mb-2 w-full origin-bottom-left
                                     bg-gray-900/95 backdrop-blur-lg rounded-lg shadow-xl
                                     ring-1 ring-black ring-opacity-5
                                     focus:outline-none overflow-hidden"
              >
                <div className="p-1">
                  <MenuItem>
                    {({ focus }) => (
                      <button
                        className={`${
                          focus ? "bg-gray-800" : ""
                        } group flex w-full items-center rounded-md px-3 py-2.5 text-sm text-white transition-colors`}
                        onClick={() => console.log("Quality settings clicked")}
                      >
                        Quality Settings
                      </button>
                    )}
                  </MenuItem>

                  <MenuItem>
                    {({ focus }) => (
                      <button
                        className={`${
                          focus ? "bg-gray-800" : ""
                        } group flex w-full items-center rounded-md px-3 py-2.5 text-sm text-white transition-colors`}
                        onClick={() => console.log("About clicked")}
                      >
                        About
                      </button>
                    )}
                  </MenuItem>
                </div>
              </MenuItems>
            </Transition>
          </Menu>
        </div>
      </div>
    </div>
  );
}

/**
 * Metric row component for consistent metric display
 */
interface MetricRowProps {
  label: string;
  value: string | number;
  unit: string;
  highlight?: boolean;
}

function MetricRow({ label, value, unit, highlight = false }: MetricRowProps) {
  return (
    <div className="flex justify-between items-baseline text-sm">
      <span className="text-gray-400">{label}</span>
      <span
        className={`font-mono font-semibold ${highlight ? "text-green-400" : "text-gray-300"}`}
      >
        {value}
        {unit && <span className="text-gray-500 ml-0.5">{unit}</span>}
      </span>
    </div>
  );
}
