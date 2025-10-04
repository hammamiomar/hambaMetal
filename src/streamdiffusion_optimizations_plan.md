# StreamDiffusion Optimizations for Music Visualizer
## Implementation Plan for M1 Max with MLX Migration Path

---

## Phase 1: Immediate Wins (PyTorch MPS - This Weekend)

### 1.1 Tiny VAE Replacement ⭐️ HIGHEST PRIORITY
**Expected speedup: 3-5x on VAE decode**

```python
from diffusers import AutoencoderTiny

viz = NoiseVisualizer()
viz.loadPipeSd()

# Single line change - massive impact
viz.pipe.vae = AutoencoderTiny.from_pretrained(
    "madebyollin/taesd",  # For SD 1.5
    torch_dtype=torch.float16
).to("mps")
```

**Why this works:**
- VAE decode is 60-70% of your pipeline time
- Tiny VAE: ~33KB latent → image (vs 200KB+ for standard VAE)
- Quality: ~95% of original, imperceptible for music viz
- No retraining needed, drop-in replacement

---

### 1.2 Increase Batch Size ⭐️ SECOND PRIORITY
**Expected speedup: 4-8x throughput**

```python
# Current
batch_size = 2  # Too small!

# Optimized
batch_size = 16  # M1 Max has 32GB+ unified memory, use it!
# Or even 32 for 1-step inference
```

**M1 Max memory budget:**
```
Standard batch size 2:
- Latents: 2 × 4 × 64 × 64 × 2 bytes = 64 KB
- Total memory: ~2GB

Optimized batch size 16:
- Latents: 16 × 4 × 64 × 64 × 2 bytes = 512 KB  
- Total memory: ~3GB (still plenty of headroom)
```

---

### 1.3 Temporal Coherence (StreamDiffusion Concept)
**Expected: Smoother visuals, same speed**

```python
class NoiseVisualizerWithTemporalCoherence(NoiseVisualizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_latent_output = None
        self.temporal_blend = 0.3  # Blend ratio
    
    def getVisualsWithTemporalBlending(self, latents, promptEmbeds, **kwargs):
        """Blend each latent with previous frame's output"""
        
        for i in range(0, num_frames, batch_size):
            batch_latents = latents[i:i+batch_size]
            
            # TEMPORAL COHERENCE: Blend first frame with previous output
            if self.prev_latent_output is not None and i == 0:
                batch_latents[0] = (
                    (1 - self.temporal_blend) * batch_latents[0] + 
                    self.temporal_blend * self.prev_latent_output
                )
            
            # Generate batch
            frames = self.pipe(...)
            
            # Store last latent for next batch
            self.prev_latent_output = batch_latents[-1].clone()
```

**Benefit:** Eliminates jarring transitions between batches

---

## Phase 2: StreamDiffusion-Inspired Features (Week 2)

### 2.1 Stochastic Similarity Filter (SSF)
**Expected: 2-3x GPU power savings during static sections**

**Your use case:** Music has quiet sections where visuals change slowly. Skip computation when similar.

```python
class MusicReactiveSimilarityFilter:
    """
    Adapted from StreamDiffusion's SSF for music visualization
    Skips frames during quiet music sections
    """
    
    def __init__(self, base_threshold=0.95):
        self.base_threshold = base_threshold
        self.prev_frame = None
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    
    def should_skip_frame(self, latent, music_energy):
        """
        Skip probability based on:
        1. Visual similarity (like StreamDiffusion)
        2. Music energy (your addition)
        """
        
        if self.prev_frame is None:
            self.prev_frame = latent.clone()
            return False
        
        # Compute similarity
        similarity = self.cos(
            self.prev_frame.reshape(-1), 
            latent.reshape(-1)
        ).item()
        
        # Adaptive threshold based on music
        # Quiet music (low energy) → higher threshold → more skipping
        # Loud music (high energy) → lower threshold → less skipping
        adaptive_threshold = self.base_threshold + (1 - music_energy) * 0.04
        
        # Probabilistic skip (StreamDiffusion's approach)
        if similarity < adaptive_threshold:
            self.prev_frame = latent.clone()
            return False
        
        skip_prob = (similarity - adaptive_threshold) / (1 - adaptive_threshold)
        skip = random.random() < skip_prob
        
        if not skip:
            self.prev_frame = latent.clone()
        
        return skip
```

**Integration:**
```python
# In your rendering loop
ssf = MusicReactiveSimilarityFilter()

for i in range(num_frames):
    latent = latents[i]
    energy = music_features[i]['energy']
    
    if ssf.should_skip_frame(latent, energy):
        # Reuse previous frame
        frames.append(prev_frame)
    else:
        # Generate new frame
        frame = pipe(latent=latent, prompt_embeds=promptEmbeds[i])
        frames.append(frame)
        prev_frame = frame
```

**Why probabilistic vs hard threshold:**
- Hard threshold causes stuttering (frame freezes)
- Probabilistic = smoother visual flow
- StreamDiffusion's key insight for streaming quality

---

### 2.2 Pre-computation Optimization (Already Doing, But Optimize More)
**StreamDiffusion pre-computes everything possible**

```python
class OptimizedNoiseVisualizer(NoiseVisualizer):
    def prepare_for_generation(self, basePrompt, targetPrompts):
        """Pre-compute EVERYTHING before generation loop"""
        
        # 1. Encode prompts (you already do this)
        self.prompt_embeds = self.getPromptEmbedsCum(...)
        
        # 2. Pre-compute noise schedule (StreamDiffusion does this)
        # For Hyper-SD, these are constant
        self.timestep = 800  # Your magic timestep
        
        # 3. Pre-allocate output tensors (avoid allocations in loop)
        self.output_frames = [None] * self.num_frames
        
        # 4. Warmup GPU (compile kernels)
        for _ in range(3):
            _ = self.pipe(
                prompt_embeds=self.prompt_embeds[0:1],
                latents=torch.randn((1, 4, 64, 64), device=self.device),
                num_inference_steps=1
            )
        
        print("Precomputation complete, GPU warmed up")
```

---

### 2.3 R-CFG (Residual Classifier-Free Guidance)
**Status: NOT APPLICABLE to your current setup**

**Why:** You're using `guidance_scale=0` (no CFG), so R-CFG doesn't help.

**If you ever need CFG:**
```python
# Instead of computing negative prompt every step (expensive):
# noise_pred = unet(latent, prompt) - unet(latent, negative_prompt)

# R-CFG uses the original input as virtual negative (cheap):
# noise_pred = unet(latent, prompt) - (latent - original_input)
```

**Skip this for now** - only relevant if you add CFG later.

---

## Phase 3: MLX Migration Strategy (Week 3-4)

### 3.1 Why MLX for M1 Max

| Feature | PyTorch MPS | MLX |
|---------|-------------|-----|
| Unified memory | Partial | Full zero-copy |
| Kernel fusion | Limited | Automatic lazy eval |
| M1 optimization | Generic | Purpose-built |
| Quantization | Manual | Built-in 4/8-bit |
| **Expected speedup** | 1x baseline | **2-3x faster** |

### 3.2 MLX Migration Path

**Step 1: Core Pipeline**
```python
# Port your core diffusion logic to MLX
import mlx.core as mx
import mlx.nn as nn

class MLXDiffusionPipeline:
    def __init__(self, model_path):
        # Load Hyper-SD weights in MLX format
        self.unet = self._load_mlx_unet(model_path)
        self.vae = self._load_mlx_tiny_vae()
        
        # Quantize immediately
        self.unet = mx.quantize(self.unet, bits=4)
    
    @mx.compile  # JIT compile for speed
    def forward(self, latent, prompt_embed):
        # Single-step Hyper-SD
        noise_pred = self.unet(latent, timestep=800, context=prompt_embed)
        denoised = latent - noise_pred  # Simplified for 1-step
        image = self.vae.decode(denoised)
        return image
```

**Step 2: Keep Your Music Analysis in NumPy**
```python
# Your librosa code stays the same
# Just convert to MLX at the boundary

# NumPy → MLX conversion
latents_np = self.getBeatLatentsCircle(...)  # Your existing code
latents_mlx = mx.array(latents_np)  # Zero-copy on M1!

# MLX → NumPy for video export
frames_mlx = mlx_pipeline.generate(latents_mlx)
frames_np = np.array(frames_mlx)  # Convert back for moviepy
```

**Step 3: Hybrid Approach (Best of Both Worlds)**
```python
class HybridMusicVisualizer:
    """
    Keep what works in PyTorch/NumPy
    Migrate only the diffusion inference to MLX
    """
    
    def __init__(self):
        # Music analysis: NumPy + librosa (FAST, no need to change)
        self.audio_analyzer = MusicAnalyzer()  # Your existing code
        
        # Prompt generation: PyTorch (fine as-is)
        from transformers import CLIPTextModel
        self.text_encoder = CLIPTextModel.from_pretrained(...)
        
        # Diffusion inference: MLX (OPTIMIZE THIS)
        self.mlx_pipeline = MLXDiffusionPipeline()
    
    def generate_video(self, audio_path):
        # 1. Analyze audio (NumPy)
        features = self.audio_analyzer.analyze(audio_path)
        
        # 2. Generate prompts (PyTorch)
        prompt_embeds = self.encode_prompts(...)
        prompt_embeds_mlx = mx.array(prompt_embeds.numpy())
        
        # 3. Generate latent walk (NumPy)
        latents = self.get_latent_walk(features)
        latents_mlx = mx.array(latents)
        
        # 4. Inference (MLX - THE FAST PART)
        frames = self.mlx_pipeline.batch_generate(latents_mlx, prompt_embeds_mlx)
        
        # 5. Export video (NumPy)
        frames_np = np.array(frames)
        create_mp4(frames_np, audio_path)
```

---

## Phase 4: Advanced Optimizations (Week 4+)

### 4.1 On-the-Fly Latent Generation
**Memory: 300MB → <1MB**

```python
def generate_visuals_streaming(self, promptEmbeds, music_features):
    """
    Don't pre-compute all latents
    Generate on-the-fly from base noise
    """
    
    # Store only base noise (tiny memory)
    shape = (1, 4, self.height//8, self.width//8)
    base_x = torch.randn(shape, device=self.device)
    base_y = torch.randn(shape, device=self.device)
    
    # Pre-compute angles (lightweight)
    angles = self._compute_circle_angles(music_features)
    
    # Generate in streaming fashion
    for i in range(num_frames):
        # Compute latent on-the-fly
        angle = angles[i]
        latent = torch.cos(angle) * base_x + torch.sin(angle) * base_y
        
        # Temporal blend with previous
        if prev_latent is not None:
            latent = 0.7 * latent + 0.3 * prev_latent
        
        # Generate frame
        frame = pipe(latent=latent, prompt_embeds=promptEmbeds[i])
        yield frame
        
        prev_latent = latent
```

---

### 4.2 Cross-Frame Attention (Advanced - Only if Multi-Step)
**Status: NOT APPLICABLE for 1-step inference**

If you ever switch to 2-4 step models:
```python
# StreamDiffusion batches frames at different denoising steps
# Frame 1 at step 0, Frame 2 at step 1, Frame 3 at step 2
# Then shares attention keys/values across frames
# Result: Future frames influence current frame (temporal consistency)
```

**Skip for now** - only works with multi-step models.

---

## Implementation Priority Ranking

### Do This Weekend (4 hours of work)
1. ✅ **Tiny VAE** (30 min) - 3-5x speedup
2. ✅ **Increase batch size to 16** (5 min) - 4-8x speedup  
3. ✅ **Temporal blending** (1 hour) - Smoother visuals
4. ✅ **Test and profile** (2 hours) - Measure improvements

**Expected result:** 0.5 FPS → 10-20 FPS at 384×384

---

### Do Week 2 (Optional, if speed not enough)
1. **Stochastic Similarity Filter** (2 hours) - 2x power savings
2. **On-the-fly latent generation** (1 hour) - 300MB memory savings
3. **Pre-computation optimization** (1 hour) - 10-20% speedup

**Expected result:** 10-20 FPS → 15-25 FPS + lower power usage

---

### Do Week 3-4 (If you want 30+ FPS)
1. **MLX pipeline port** (1 week) - 2-3x speedup
2. **4-bit quantization** (already built into MLX) - Another 1.5-2x
3. **Metal kernel optimization** (optional, advanced) - 1.5-2x more

**Expected result:** 15-25 FPS → 30-50 FPS at 384×384

---

## Final Architecture Comparison

### Current (PyTorch MPS)
```
Audio → librosa analysis
     → Prompt encoding (PyTorch)
     → Latent walk generation (NumPy)
     → [PRE-COMPUTE ALL LATENTS - 300MB]
     → Batch diffusion (PyTorch, batch=2)
     → VAE decode (SLOW)
     → Video export

Speed: 0.5-1 FPS at 512×512
Memory: 2-3GB
```

### Optimized PyTorch (Weekend Implementation)
```
Audio → librosa analysis  
     → Prompt encoding (PyTorch, CACHED)
     → On-the-fly latent generation (<1MB)
     → Batch diffusion (PyTorch, batch=16)
     → Tiny VAE decode (FAST)
     → Temporal blending
     → Similarity filter (optional)
     → Video export

Speed: 15-25 FPS at 384×384
Memory: 1-2GB
```

### Ultimate MLX (Week 3-4 Implementation)
```
Audio → librosa analysis
     → Prompt encoding (PyTorch, CACHED)
     → On-the-fly latent generation (<1MB)
     → Batch diffusion (MLX, batch=32, 4-bit quantized)
     → Tiny VAE decode (MLX, compiled)
     → Temporal blending
     → Similarity filter
     → Video export

Speed: 30-50 FPS at 384×384 (or 15-25 FPS at 512×512)
Memory: 1GB
Power: 30-40% less than current
```

---

## Code Migration Checklist

### Phase 1: Immediate (PyTorch MPS)
- [ ] Replace VAE with Tiny VAE
- [ ] Change batch_size from 2 to 16
- [ ] Add temporal blending to getVisuals()
- [ ] Profile and measure speedup
- [ ] Test video quality

### Phase 2: Enhancements
- [ ] Implement SSF with music-reactive thresholds
- [ ] Switch to on-the-fly latent generation
- [ ] Add pre-computation warmup
- [ ] Profile power usage

### Phase 3: MLX Migration
- [ ] Research MLX Stable Diffusion implementations
- [ ] Port UNet to MLX with 4-bit quantization
- [ ] Port Tiny VAE to MLX
- [ ] Keep music analysis in NumPy
- [ ] Benchmark MLX vs PyTorch
- [ ] Implement hybrid pipeline

### Phase 4: Polish
- [ ] Fine-tune temporal blending ratio
- [ ] Optimize SSF thresholds for your music
- [ ] Add beat-synchronized effects
- [ ] Render final videos

---

## Key Takeaways from StreamDiffusion

1. **Temporal coherence > raw speed** - Smooth visuals matter more than FPS
2. **Tiny VAE is the biggest win** - 3-5x speedup for minimal quality loss
3. **Batch everything possible** - Don't waste unified memory on M1
4. **Pre-compute aggressively** - Do work once, reuse everywhere
5. **Probabilistic skipping > hard thresholds** - SSF is brilliant for smooth streaming
6. **MLX is purpose-built for M1** - 2-3x faster than PyTorch MPS

**Your advantage:** Music visualization is offline rendering, not real-time streaming. You can:
- Use bigger batch sizes (no latency constraints)
- Pre-compute more aggressively
- Trade memory for speed
- Run multi-pass if needed

**You don't need** full StreamDiffusion complexity:
- R-CFG (you use guidance_scale=0)
- Cross-frame attention (1-step inference)
- Input/Output queues (offline rendering)

**Start simple, optimize incrementally.**
