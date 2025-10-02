# StreamDiffusion Music Visualizer - Frontend

A high-performance React frontend for visualizing and profiling StreamDiffusion inference with Metal Performance Shaders (MPS).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         App.tsx                             │
│  - Main composition layer                                    │
│  - Manages WebSocket connection                              │
│  - Coordinates Canvas and Controls                           │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────┐      ┌──────────────┐     ┌──────────────┐
│   Canvas    │      │  Controls    │     │ useWebSocket │
│             │      │              │     │    Hook      │
│ Imperative  │      │  Headless    │     │              │
│ rendering   │      │  UI + Glass  │     │  useRef      │
│ outside     │      │  morphism    │     │  pattern     │
│ React       │      │              │     │              │
└─────────────┘      └──────────────┘     └──────────────┘
```

## File Structure

```
src/
├── components/
│   ├── Canvas.tsx         # High-perf canvas rendering (forwardRef + imperative)
│   ├── Controls.tsx       # UI overlay with Headless UI components
│   └── ErrorBoundary.tsx  # Error catching and fallback UI
├── hooks/
│   └── useWebSocket.ts    # WebSocket management with useRef pattern
├── types.ts               # TypeScript type definitions
├── constants.ts           # Configuration constants
├── mps-integration.md     # Guide for backend integration
├── App.tsx                # Main application component
├── main.tsx               # React entry point
└── index.css              # Tailwind CSS imports
```

## Key React Patterns Used

### 1. **Imperative Rendering with `forwardRef` + `useImperativeHandle`**

**File**: `components/Canvas.tsx`

**Why?** Rendering at 30-60 FPS through React's reconciliation would add latency. By exposing an imperative API, we bypass React and draw directly to the canvas.

```typescript
export const Canvas = forwardRef<CanvasHandle, CanvasProps>((props, ref) => {
  useImperativeHandle(ref, () => ({
    renderFrame: async (data: ArrayBuffer) => {
      // Direct canvas rendering without React re-render
    },
  }));
});

// Usage in parent:
canvasRef.current?.renderFrame(frameData);
```

**Learning**: Use this pattern when performance requires bypassing React's declarative model (rare!).

---

### 2. **The "Latest Ref" Pattern (Critical!)**

**File**: `hooks/useWebSocket.ts`

**The Problem**: When `App` re-renders (e.g., FPS updates), the `onFrame` callback is recreated. If `onFrame` is in the dependency array of `useCallback`, it causes the WebSocket to disconnect/reconnect.

**The Solution**: Store the callback in a `useRef` and update it on every render without triggering effects.

```typescript
// Store latest callback in ref
const onFrameRef = useRef(onFrame);

// Update ref on every render (no effect re-run!)
useEffect(() => {
  onFrameRef.current = onFrame;
});

// Use ref in WebSocket message handler
ws.current.onmessage = (event) => {
  onFrameRef.current?.(event.data);
};

// connect() only depends on url, not onFrame
const connect = useCallback(() => {
  // ... setup WebSocket ...
}, [url]); // onFrame NOT in dependencies!
```

**Learning**: When you need the latest value but don't want to trigger cleanup/re-run, use a ref.

---

### 3. **Component Composition**

**File**: `App.tsx`

Small, focused components combined to build the full application:

- **Canvas**: Rendering only
- **Controls**: UI only (presentational component)
- **useWebSocket**: Connection logic only
- **App**: Composes everything together

**Learning**: Break complex UIs into simple, reusable pieces. Each component should do one thing well.

---

### 4. **Error Boundaries**

**File**: `components/ErrorBoundary.tsx`

Class component that catches errors in child components and shows fallback UI.

```typescript
<ErrorBoundary>
  <App />
</ErrorBoundary>
```

**Learning**: Always wrap major sections of your app. Prevents one component crash from breaking the entire app.

---

### 5. **Custom Hooks**

**File**: `hooks/useWebSocket.ts`

Encapsulates stateful logic for reuse across components.

```typescript
const { connect, disconnect, status, fps } = useWebSocket({
  url: WS_CONFIG.URL,
  onFrame: handleFrame,
});
```

**Learning**: Extract complex stateful logic into custom hooks. Makes components simpler and logic testable.

---

## Performance Optimizations

### Browser APIs

1. **`createImageBitmap()`** - Hardware-accelerated image decoding
2. **`desynchronized: true`** - Reduces latency between draw and screen update
3. **`alpha: false`** - Skips alpha channel processing

### React Patterns

1. **Imperative canvas rendering** - Bypasses reconciliation
2. **`useCallback`** - Stable function references
3. **useRef pattern** - Prevents unnecessary effect re-runs

### WebSocket

1. **Binary frames** - No JSON overhead for images
2. **Immediate rendering** - No buffering
3. **Automatic reconnection** - Resilient to network issues

---

## Running the Application

### Development

```bash
# Install dependencies (if not done)
npm install

# Start dev server (port 3000)
npm run dev

# Start backend (port 8080)
cd ../..
uvicorn app.main:app --reload --port 8080
```

### Testing WebSocket Connection

1. Start backend: `uvicorn app.main:app --reload --port 8080`
2. Start frontend: `npm run dev`
3. Open http://localhost:3000
4. Click "Connect" in the UI
5. You should see random frames rendering at ~30 FPS

### Debugging

**WebSocket not connecting?**
- Check backend is running on port 8080
- Check browser console for errors
- Verify Vite proxy config in `vite.config.ts`

**Frames not rendering?**
- Open browser DevTools → Network → WS tab
- Verify binary frames are being received
- Check Canvas errors in console

**Performance issues?**
- Check FPS counter in Controls panel
- Use Chrome DevTools → Performance tab
- Verify `createImageBitmap` is being used (check Canvas.tsx)

---

## Next Steps

### Immediate
- [x] Basic WebSocket streaming (done!)
- [ ] Test with backend running
- [ ] Verify FPS counter works

### StreamDiffusion Integration
- [ ] Add StreamDiffusion pipeline to backend
- [ ] Send MPS metrics via WebSocket (see `mps-integration.md`)
- [ ] Display inference time in Controls
- [ ] Add prompt input for txt2img

### Advanced Features
- [ ] Latency measurement (ping/pong)
- [ ] Frame quality controls
- [ ] Recording/screenshot functionality
- [ ] Multiple stream sources
- [ ] Audio reactivity for music visualizer

---

## Learning Resources

**React Patterns**:
- [React Beta Docs](https://react.dev/) - Modern React with hooks
- [Patterns.dev](https://www.patterns.dev/posts/reactpatterns/) - Advanced patterns

**Performance**:
- [React Performance](https://react.dev/learn/render-and-commit) - Understanding renders
- [Web Performance](https://web.dev/performance/) - Browser optimization

**TypeScript**:
- [React TypeScript Cheatsheet](https://react-typescript-cheatsheet.netlify.app/) - Best practices

**Headless UI**:
- [Official Docs](https://headlessui.com/) - Accessible components

---

## Code Quality

**ESLint**: Linting configured with React hooks rules
**TypeScript**: Strict mode for type safety
**Comments**: Extensive inline documentation for learning
**Architecture**: Clean separation of concerns

---

## Questions?

The code is heavily commented to teach idiomatic React patterns. If you're unsure about something:

1. Read the comments in the relevant file
2. Check this README
3. Refer to `mps-integration.md` for backend integration

Built to learn React the right way. Every pattern used here is production-ready and follows current best practices. 🚀
