# hambaMetal App - Functionality Overview

## Architecture

Real-time video streaming application with FastAPI backend and React frontend, designed for low-latency frame delivery over WebSockets.

## Backend (FastAPI + Python)

### Main Server (`app/main.py`)
- FastAPI application with CORS middleware (allows all origins)
- Runs on `localhost:8080`
- Includes diffusion router for streaming endpoints

### Diffusion Router (`app/routers/diffusion.py`)

**Endpoints:**

1. **GET `/test/random`**
   - Returns a single 512x512 random noise image as JPEG
   - Uses OpenCV for RGB→BGR conversion and JPEG encoding (90% quality)

2. **WebSocket `/ws/test/`**
   - Streams random 512x512 noise images at 30 FPS
   - Binary WebSocket (ArrayBuffer) for low-latency frame delivery
   - Frame timing controlled via `asyncio.sleep()` to maintain target FPS
   - Handles disconnects and errors gracefully

**Image Processing:**
- `array_to_buffer()`: Converts NumPy arrays to JPEG bytes
  - Auto-scales values if needed (0-1 range → 0-255)
  - RGB to BGR conversion for OpenCV compatibility
  - JPEG compression at 90% quality

## Frontend (React + TypeScript + Vite)

### Core Components

**App.tsx** (Main orchestrator)
- Manages WebSocket connection via `useWebSocket` hook
- Refs Canvas component for imperative frame rendering
- Aggregates metrics (FPS, future: inference time, latency)
- Renders full-screen black canvas with overlay controls

**Canvas.tsx** (Hardware-accelerated rendering)
- Exposes imperative `CanvasHandle` API via `useImperativeHandle`
  - `renderFrame(data: ArrayBuffer)`: Decode and render JPEG frame
  - `clear()`: Fill canvas with black
- Uses `createImageBitmap()` for GPU-accelerated JPEG decoding
- Implements "cover" scaling (fills canvas, maintains aspect ratio, centers image)
- Critical memory management: calls `img.close()` to prevent bitmap leaks
- Canvas context optimizations:
  - `alpha: false` - skip transparency processing
  - `desynchronized: true` - lower latency, allows tearing (invisible at 30fps)
- Auto-resizes canvas buffer to match CSS dimensions on window resize

**Controls.tsx** (Draggable UI panel)
- Glassmorphic draggable panel (pointer capture API)
- Connection status indicator with pulsing animation when connected
- Connect/Disconnect buttons (disabled based on state)
- Real-time metrics display:
  - Client FPS (measured from WebSocket frame rate)
  - Inference time (planned, not yet implemented)
  - Latency (planned, not yet implemented)
- Settings dropdown menu (UI only, not functional yet)
- Sage green color theme (`#8B9A7E`, `#B5CC9A`, `#9CA986`)

**ErrorBoundary.tsx**
- React error boundary for graceful error handling

### Custom Hooks

**useWebSocket.ts** (WebSocket management)
- Manages WebSocket lifecycle with connection status tracking
- Features:
  - Auto-connect on mount (optional)
  - Auto-reconnect with exponential backoff (configurable)
  - FPS calculation (counts frames, updates every 1 second)
  - Defensive callback handling (stores `onFrame` in ref to prevent reconnects on parent re-renders)
- Status states: `DISCONNECTED`, `CONNECTING`, `CONNECTED`, `ERROR`
- Handles both binary (ArrayBuffer) and JSON messages
- Reconnection logic:
  - Max attempts configurable via `WS_CONFIG.MAX_RECONNECT_ATTEMPTS`
  - Delay configurable via `WS_CONFIG.RECONNECT_DELAY`
  - Skips reconnect on normal closure (code 1000)

### Types (`types.ts`)

**ConnectionStatus**
- `disconnected | connecting | connected | error`

**Metrics Interfaces**
- `BackendMetrics`: inference time, frame dimensions (planned)
- `ClientMetrics`: FPS, WebSocket latency
- `Metrics`: Combined client + backend metrics

**WebSocketMessage** (Discriminated union)
- `{ type: 'metrics', data: BackendMetrics }`
- `{ type: 'frame', data: ArrayBuffer }`

### Configuration (`constants.ts`)

**WS_CONFIG**
- WebSocket URL, reconnect settings

**PERF_CONFIG**
- FPS update interval (1000ms)

**CANVAS_CONFIG**
- Background color (`#000000`)

## Performance Optimizations

### Backend
- JPEG encoding at 90% quality (balance size/quality)
- Target 30 FPS with frame timing compensation
- Binary WebSocket for minimal overhead

### Frontend
- Hardware-accelerated JPEG decoding (`createImageBitmap`)
- Canvas context flags for performance (`alpha: false`, `desynchronized: true`)
- Manual bitmap disposal to prevent memory leaks
- Defensive refs to prevent unnecessary WebSocket reconnections
- Minimal re-renders (status/metrics stored in state, callbacks in refs)

## Future Integration Notes

Comments indicate planned integration with **StreamDiffusion**:
- Backend will send inference metrics via JSON messages
- Frontend already has metric display placeholders
- WebSocket handler supports both binary frames and JSON metadata

## Development Stack

**Backend:**
- FastAPI (async Python web framework)
- NumPy (array processing)
- OpenCV (cv2, image encoding)
- Uvicorn (ASGI server)

**Frontend:**
- React 18 (UI framework)
- TypeScript (type safety)
- Vite (build tool)
- Headless UI (accessible components)
- TailwindCSS (utility-first CSS)

## Running the App

**Backend:**
```bash
python -m app.main
# or
uvicorn app.main:app --host localhost --port 8080 --reload
```

**Frontend:**
```bash
cd app/frontend
npm install
npm run dev
```

**Connect:**
1. Open frontend in browser
2. Click "Connect" in control panel
3. Watch random noise stream at 30 FPS
