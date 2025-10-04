# hambaMetal - Refactored Architecture

clean, type-safe, modular codebase for real-time music visualization using diffusion models

## architecture overview

```
src/
├── config.py              # pydantic configs (type-safe, validated)
├── audio/                 # audio feature extraction
│   ├── analyzer.py        # librosa-based analysis
│   └── types.py           # immutable audio feature containers
├── diffusion/             # diffusion pipelines
│   ├── base.py            # abstract base pipeline
│   ├── sd15.py            # sd1.5 implementation (hyper-sd)
│   └── stream.py          # streamdiffusion-inspired optimizations
├── latent/                # latent generation
│   └── generator.py       # music-synchronized latent walks
├── prompt/                # prompt interpolation
│   └── interpolator.py    # chroma-based prompt blending
└── profiling/             # performance analysis
    └── mps_profiler.py    # mps/cuda timing and memory tracking
```

---

## core concepts

### 1. pydantic configs (config.py)

all configuration is type-safe using pydantic models:

```python
from src.config import PipelineConfig

# use preset configs
config = PipelineConfig.fast_preset()  # optimized for speed
config = PipelineConfig.balanced_preset()  # balance speed/quality
config = PipelineConfig.quality_preset()  # max quality

# or customize
config = PipelineConfig(
    diffusion=DiffusionConfig(
        use_tiny_vae=True,  # 3-5x speedup
        device="mps",
    ),
    stream=StreamConfig(
        enable_temporal_coherence=True,
        temporal_blend_ratio=0.3,
    ),
    inference=InferenceConfig(
        batch_size=16,
        height=384,
        width=384,
    ),
)
```

### 2. audio analysis (audio/)

extract musical features from audio files:

```python
from src.audio import AudioAnalyzer, AudioFeatures

analyzer = AudioAnalyzer(config.audio)
features: AudioFeatures = analyzer.analyze("song.mp3")

# features is immutable (frozen dataclass)
print(features.tempo)  # bpm
print(features.num_frames)
print(features.beat_frames)  # beat locations
print(features.chroma_cq)  # chromagram (12, num_frames)

# query features
energy = features.get_energy_at_frame(100)
is_beat = features.is_beat_frame(100)
chroma = features.get_chroma_at_frame(100)  # 12-dim vector
```

### 3. diffusion pipelines (diffusion/)

flexible, model-agnostic interface:

```python
from src.diffusion import SD15Pipeline, StreamDiffusionPipeline

# create base pipeline (sd1.5, sdxl, flux, etc.)
base = SD15Pipeline(
    diffusion_config=config.diffusion,
    inference_config=config.inference,
)

# wrap with streaming optimizations
pipeline = StreamDiffusionPipeline(
    base_pipeline=base,
    stream_config=config.stream,
)

# encode prompts
prompt_embeds = pipeline.encode_prompt("abstract flowing shapes")

# generate single frame
latent = torch.randn(pipeline.get_latent_shape())
frame = pipeline.generate_frame_with_coherence(
    latent=latent,
    prompt_embeds=prompt_embeds,
    music_energy=0.8,  # current music energy [0, 1]
)

# or batch generation
frames = pipeline.base.generate_batch(latents, prompt_embeds)
```

### 4. latent generation (latent/)

music-synchronized latent walks:

```python
from src.latent import LatentGenerator

latent_gen = LatentGenerator(
    config=config.latent,
    device="mps",
    dtype=torch.float16,
)

# precompute all latents
latents = latent_gen.generate_latent_walk(
    audio_features=features,
    latent_shape=(1, 4, 64, 64),
    precompute=True,  # generate all at once
)

# or on-the-fly (memory efficient)
latents = latent_gen.generate_latent_walk(
    audio_features=features,
    latent_shape=(1, 4, 64, 64),
    precompute=False,  # placeholder
)

# then generate frame-by-frame
for i in range(features.num_frames):
    latent = latent_gen.get_latent_at_frame(i, features)
    # use latent...
```

### 5. prompt interpolation (prompt/)

chroma-based prompt blending:

```python
from src.prompt import PromptInterpolator

prompt_interp = PromptInterpolator(
    config=config.prompt,
    device="mps",
    dtype=torch.float16,
)

# encode base and chroma prompts
base_embeds = pipeline.encode_prompt("abstract art")
chroma_embeds = pipeline.encode_prompt([
    "red warm",
    "orange vibrant",
    # ... 12 prompts for chromatic scale
])

# interpolate based on music
prompt_embeds = prompt_interp.interpolate_prompts(
    audio_features=features,
    base_prompt_embeds=base_embeds,
    chroma_prompt_embeds=chroma_embeds,
)
# returns (num_frames, seq_len, hidden_size)
```

### 6. profiling (profiling/)

track performance bottlenecks:

```python
from src.profiling import get_profiler

profiler = get_profiler()

# profile code blocks
with profiler.profile("unet_forward"):
    output = unet(input)

with profiler.profile("vae_decode"):
    image = vae.decode(latent)

# view results
profiler.print_summary()
profiler.export_json("profile.json")

# get specific metrics
avg_unet_time = profiler.get_avg_time("unet_forward")
```

---

## streamdiffusion optimizations

### temporal coherence

smooth frame-to-frame transitions by blending with previous frame:

```python
config.stream.enable_temporal_coherence = True
config.stream.temporal_blend_ratio = 0.3  # 30% previous, 70% current

# adaptive blending based on music energy
# high energy (loud) -> less blending (more responsive)
# low energy (quiet) -> more blending (smoother)
```

### tiny vae

replace standard vae with tiny vae for 3-5x decode speedup:

```python
config.diffusion.use_tiny_vae = True
# ~95% quality of standard vae, imperceptible for music viz
```

### stochastic similarity filter

skip similar frames during quiet sections (optional):

```python
config.stream.enable_similarity_filter = True
config.stream.similarity_threshold = 0.95

# probabilistic skipping (no hard cutoffs)
# adapts to music energy
```

### batch size optimization

m1 max has 32gb+ unified memory - use it!

```python
config.inference.batch_size = 16  # or even 32 for 1-step
# old code: batch_size = 2 (way too small)
```

---

## usage examples

### offline video generation

```python
from src.config import PipelineConfig
from src.audio import AudioAnalyzer
from src.diffusion import SD15Pipeline, StreamDiffusionPipeline
from src.latent import LatentGenerator
from src.prompt import PromptInterpolator

# setup
config = PipelineConfig.balanced_preset()
analyzer = AudioAnalyzer(config.audio)
features = analyzer.analyze("song.mp3")

# create pipeline
base = SD15Pipeline(config.diffusion, config.inference)
pipeline = StreamDiffusionPipeline(base, config.stream)

# generate latents and prompts
latent_gen = LatentGenerator(config.latent, "mps", torch.float16)
latents = latent_gen.generate_latent_walk(
    features, pipeline.get_latent_shape(), precompute=True
)

prompt_interp = PromptInterpolator(config.prompt, "mps", torch.float16)
base_embeds = pipeline.encode_prompt(config.prompt.base_prompt)
chroma_embeds = pipeline.encode_prompt(config.prompt.chroma_prompts)
prompt_embeds = prompt_interp.interpolate_prompts(
    features, base_embeds[0], chroma_embeds
)

# generate all frames
frames = pipeline.base.generate_batch(latents, prompt_embeds)

# export to video (use moviepy or similar)
```

### real-time streaming

```python
# setup pipeline (same as above)

# on-the-fly generation
for frame_idx in range(features.num_frames):
    # get latent for this frame
    latent = latent_gen.get_latent_at_frame(frame_idx, features)

    # get energy for adaptive blending
    energy = features.get_energy_at_frame(frame_idx)

    # generate with temporal coherence
    frame = pipeline.generate_frame_with_coherence(
        latent=latent,
        prompt_embeds=prompt_embeds[frame_idx:frame_idx+1],
        music_energy=energy,
    )

    # stream frame (websocket, etc)
    send_frame(frame)
```

### fastapi integration

see `app/routers/diffusion.py` for websocket endpoint example

---

## key differences from old code

### old (imagegen.py)
- monolithic `NoiseVisualizer` class (1300+ lines)
- no type safety (dicts, optional args everywhere)
- tight coupling (audio, diffusion, latents all mixed)
- manual memory management
- no profiling
- hard to swap models

### new (src/)
- modular, single-responsibility classes
- pydantic configs with validation
- clear separation of concerns
- type hints everywhere (rust-like safety)
- built-in profiling
- model-agnostic interface (sd15, sdxl, flux, etc.)
- streamdiffusion optimizations
- memory efficient (on-the-fly generation)

---

## extending the pipeline

### add new model (e.g., flux)

```python
# src/diffusion/flux.py
from .base import BaseDiffusionPipeline

class FluxPipeline(BaseDiffusionPipeline):
    def load_model(self):
        # flux-specific loading
        pass

    def encode_prompt(self, prompt):
        # flux-specific encoding
        pass

    def supports_pooled_embeds(self):
        return False  # or True for flux
```

### add new latent generation mode

```python
# src/latent/generator.py
class LatentGenerator:
    def generate_spiral_walk(self, audio_features, ...):
        # new latent motion pattern
        pass
```

### add new prompt interpolation method

```python
# src/prompt/interpolator.py
class PromptInterpolator:
    def _interpolate_custom(self, ...):
        # new interpolation algorithm
        pass
```

---

## performance tips

### for m1 max (mps)

```python
# use fast preset
config = PipelineConfig.fast_preset()

# tiny vae (critical!)
config.diffusion.use_tiny_vae = True

# large batch size
config.inference.batch_size = 16  # or 32

# lower resolution for real-time
config.inference.height = 384
config.inference.width = 384

# temporal coherence for smooth visuals
config.stream.enable_temporal_coherence = True
config.stream.temporal_blend_ratio = 0.3

# expected: 10-20 fps at 384x384 on m1 max
```

### for cuda

```python
# enable optimizations
config.diffusion.compile_unet = True
config.diffusion.enable_channels_last = True

# expected: 30-50 fps at 512x512 on rtx 3090
```

---

## todo / future work

- [ ] mlx backend for m1 max (2-3x faster than mps)
- [ ] multi-resolution generation (start low, upscale)
- [ ] audio-reactive controlnet (dance videos)
- [ ] real-time audio input (microphone)
- [ ] video export utilities
- [ ] web ui for parameter tuning
- [ ] batch processing multiple songs

---

## development notes

### code style
- lowercase comments ("blend previous frame")
- type hints everywhere
- pydantic for configs
- clear, concise naming
- separation of concerns
- rusty vibes (explicit > implicit)

### testing
```bash
# run demo
python demo_usage.py

# start fastapi server
python -m app.main

# connect frontend
cd app/frontend && npm run dev
```

### profiling
```bash
# profiling automatically enabled
# results in profiling_results.json
# or call profiler.print_summary()
```

---

## credits

original hambaJubaTuba by @omarhammami
refactored with streamdiffusion inspiration
