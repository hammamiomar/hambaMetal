# MPS Profiling Integration Guide

This document outlines how to integrate Metal Performance Shaders (MPS) profiling metrics from the StreamDiffusion backend into the frontend.

## Current Status

✅ **Frontend Ready**: Types and UI components are prepared for metrics display
⏳ **Backend TODO**: Integrate StreamDiffusion pipeline and send metrics via WebSocket

## Backend Integration Steps

### 1. Update Backend to Send Metrics + Frames

Modify `app/routers/diffusion.py` to send both metrics (JSON) and frames (binary):

```python
@router.websocket("/ws/stream")
async def stream_diffusion(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to StreamDiffusion")

    # Initialize StreamDiffusion pipeline
    # stream = StreamDiffusion(...)

    try:
        while True:
            # MPS profiling (from pipeline.py lines 428-460)
            start = torch.mps.Event(enable_timing=True)
            end = torch.mps.Event(enable_timing=True)
            start.record()

            # Generate frame
            output_tensor = stream(input_frame)  # Returns (1, 3, H, W) on MPS

            end.record()
            torch.mps.synchronize()  # CRITICAL: Wait for GPU
            inference_ms = start.elapsed_time(end)

            # Send metrics as JSON
            await websocket.send_json({
                "type": "metrics",
                "data": {
                    "inferenceTimeMs": inference_ms,
                    "inferenceTimeEmaMs": stream.inference_time_ema * 1000,
                }
            })

            # Convert tensor to JPEG bytes
            frame_bytes = tensor_to_buffer(output_tensor)

            # Send frame as binary
            await websocket.send_bytes(frame_bytes)

    except WebSocketDisconnect:
        logger.info("Client disconnected")
```

### 2. MPS Tensor to JPEG Conversion

Add this helper function to properly convert MPS tensors:

```python
def tensor_to_buffer(tensor: torch.Tensor) -> bytes:
    """
    Convert StreamDiffusion output tensor to JPEG bytes

    Args:
        tensor: (1, 3, H, W) tensor on MPS device, range [-1, 1] or [0, 1]

    Returns:
        JPEG bytes ready for WebSocket transmission
    """
    # Move to CPU first (MPS tensors don't auto-transfer)
    img = tensor.detach().cpu().squeeze(0)  # (3, H, W)
    img = img.permute(1, 2, 0).numpy()      # (H, W, 3)

    # Normalize to [0, 255]
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.clip(0, 255).astype(np.uint8)

    # RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Encode as JPEG
    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()
```

## Frontend Updates (Already Done!)

The frontend is already configured to handle metrics. The `useWebSocket` hook automatically processes JSON messages:

```typescript
// In useWebSocket.ts (already implemented)
ws.current.onmessage = (event: MessageEvent) => {
  if (typeof event.data === 'string') {
    const metadata = JSON.parse(event.data);
    // TODO: Update state with metrics
  } else if (event.data instanceof ArrayBuffer) {
    onFrameRef.current?.(event.data);
  }
};
```

### Next Step: Add Metrics State

When you're ready, update `useWebSocket.ts` to expose metrics:

```typescript
const [backendMetrics, setBackendMetrics] = useState<BackendMetrics>({});

// In onmessage handler:
if (typeof event.data === 'string') {
  const msg = JSON.parse(event.data);
  if (msg.type === 'metrics') {
    setBackendMetrics(msg.data);
  }
}

// Return metrics
return {
  connect,
  disconnect,
  status,
  fps,
  reconnectAttempts,
  backendMetrics, // Add this
};
```

Then update `App.tsx`:

```typescript
const { ..., backendMetrics } = useWebSocket({...});

const metrics: Metrics = {
  fps,
  ...backendMetrics, // Merge backend metrics
};
```

## MPS-Specific Considerations

### Synchronization is Critical
```python
# ALWAYS synchronize before reading timing!
torch.mps.synchronize()
inference_time = start.elapsed_time(end) / 1000
```

Without `synchronize()`, you'll get incorrect timings because MPS operations are async.

### Memory Management
```python
# Explicitly move to CPU
tensor.detach().cpu()

# MPS tensors don't auto-transfer like CUDA
```

### Profiling Tools

**For detailed GPU profiling on macOS:**
1. **Xcode Instruments** - Metal System Trace
2. **Activity Monitor** - GPU History tab
3. **`torch.mps.profiler`** (experimental)

**Example with torch profiler:**
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.MPS],
    record_shapes=True
) as prof:
    output = stream(input_frame)

print(prof.key_averages().table(sort_by="mps_time_total"))
```

## Additional Metrics to Consider

Expand `BackendMetrics` type in `types.ts`:

```typescript
export interface BackendMetrics {
  // Timing
  inferenceTimeMs?: number;
  inferenceTimeEmaMs?: number;

  // Throughput
  throughputFps?: number;

  // Quality
  denosingSteps?: number;

  // Dimensions
  frameWidth?: number;
  frameHeight?: number;

  // Model info
  modelName?: string;
  guidanceScale?: number;
}
```

## Testing

1. **Test random frames first**: Current `/ws/test/` endpoint
2. **Add metrics to test endpoint**: Verify frontend handles JSON + binary
3. **Integrate StreamDiffusion**: Replace random frames with real diffusion
4. **Profile performance**: Use Instruments to verify MPS utilization

## Resources

- StreamDiffusion pipeline: `StreamDiffusion/src/streamdiffusion/pipeline.py`
- MPS events API: https://pytorch.org/docs/stable/notes/mps.html
- WebSocket binary frames: https://developer.mozilla.org/en-US/docs/Web/API/WebSocket
