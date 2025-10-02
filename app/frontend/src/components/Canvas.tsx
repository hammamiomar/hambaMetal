import { useEffect, useRef, forwardRef, useImperativeHandle } from 'react';
import { CANVAS_CONFIG } from '../constants';

/**
 * Props for the Canvas component
 */
interface CanvasProps {
  /** Additional CSS classes */
  className?: string;
}

/**
 * Imperative handle exposed by the Canvas component
 * This allows parent components to render frames without triggering React re-renders
 */
export interface CanvasHandle {
  /**
   * Render a frame to the canvas
   * @param data - ArrayBuffer containing JPEG image data
   */
  renderFrame: (data: ArrayBuffer) => Promise<void>;

  /**
   * Clear the canvas to black
   */
  clear: () => void;
}

/**
 * High-performance canvas component for rendering video frames
 *
 * ARCHITECTURE NOTES:
 * - Uses forwardRef + useImperativeHandle for imperative API
 * - Rendering happens OUTSIDE React's render cycle
 * - This is critical for performance at 30-60 FPS
 *
 * WHY THIS PATTERN?
 * - Calling setState() 60 times/second would trigger React reconciliation
 * - React would diff the virtual DOM, update fiber tree, etc.
 * - This adds ~5-10ms latency per frame
 * - Instead, we bypass React and draw directly to the canvas
 *
 * LEARNING REACT:
 * - forwardRef: Allows parent to get a reference to this component
 * - useImperativeHandle: Customizes the value exposed via that ref
 * - This is an "escape hatch" for imperative operations
 * - Use sparingly! Only when React's declarative model doesn't fit
 *
 * BROWSER OPTIMIZATIONS:
 * - createImageBitmap(): Hardware-accelerated, runs on GPU/compositor thread
 * - desynchronized: true: Reduces latency between drawImage() and screen update
 * - alpha: false: Skips alpha channel processing (performance gain)
 *
 * @param props - Component props
 * @param ref - Forwarded ref that will receive the CanvasHandle
 */
export const Canvas = forwardRef<CanvasHandle, CanvasProps>(({ className }, ref) => {
  // DOM reference to the canvas element
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // 2D rendering context (stored to avoid repeated getContext calls)
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);

  /**
   * Initialize canvas context and handle window resize
   */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Get rendering context with performance optimizations
    ctxRef.current = canvas.getContext('2d', {
      alpha: false,        // No transparency = faster compositing
      desynchronized: true, // Lower latency, allow tearing (imperceptible at high FPS)
    });

    /**
     * Resize canvas to match display size
     * Important: Canvas has TWO sizes:
     * 1. Display size (CSS width/height) - how big it looks
     * 2. Drawing buffer size (canvas.width/height) - resolution
     * They should match for crisp rendering
     */
    const handleResize = () => {
      if (!canvas) return;

      const { clientWidth, clientHeight } = canvas;

      // Only resize if dimensions changed (avoid unnecessary clears)
      if (canvas.width !== clientWidth || canvas.height !== clientHeight) {
        canvas.width = clientWidth;
        canvas.height = clientHeight;

        // Fill with black after resize
        if (ctxRef.current) {
          ctxRef.current.fillStyle = CANVAS_CONFIG.BACKGROUND_COLOR;
          ctxRef.current.fillRect(0, 0, clientWidth, clientHeight);
        }
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  /**
   * Expose imperative methods to parent via ref
   */
  useImperativeHandle(ref, () => ({
    /**
     * Render a frame from ArrayBuffer (JPEG data)
     */
    renderFrame: async (data: ArrayBuffer) => {
      const ctx = ctxRef.current;
      const canvas = canvasRef.current;
      if (!ctx || !canvas) return;

      try {
        // Convert ArrayBuffer to Blob
        const blob = new Blob([data], { type: 'image/jpeg' });

        // Decode to ImageBitmap (fastest method, uses hardware acceleration)
        // This happens off the main thread in modern browsers
        const img = await createImageBitmap(blob);

        // Calculate scaling to fill canvas while maintaining aspect ratio
        // (cover behavior, like CSS background-size: cover)
        const scale = Math.max(
          canvas.width / img.width,
          canvas.height / img.height
        );

        const scaledWidth = img.width * scale;
        const scaledHeight = img.height * scale;

        // Center the image
        const x = (canvas.width - scaledWidth) / 2;
        const y = (canvas.height - scaledHeight) / 2;

        // Clear and draw in one pass (no flicker)
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, x, y, scaledWidth, scaledHeight);

        // Cleanup: ImageBitmap must be manually closed
        // (not garbage collected automatically)
        img.close();
      } catch (error) {
        console.error('[Canvas] Error rendering frame:', error);
      }
    },

    /**
     * Clear canvas to black
     */
    clear: () => {
      const ctx = ctxRef.current;
      const canvas = canvasRef.current;
      if (!ctx || !canvas) return;

      ctx.fillStyle = CANVAS_CONFIG.BACKGROUND_COLOR;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    },
  }));

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{
        display: 'block',
        width: '100%',
        height: '100%',
      }}
    />
  );
});

// Display name for React DevTools
Canvas.displayName = 'Canvas';
