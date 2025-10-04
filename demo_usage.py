"""
demo usage of refactored hambaMetal pipeline
shows how to use all the new modules together
"""

import torch
from pathlib import Path

from src.config import PipelineConfig
from src.audio import AudioAnalyzer
from src.diffusion import SD15Pipeline, StreamDiffusionPipeline
from src.latent import LatentGenerator
from src.prompt import PromptInterpolator
from src.profiling import get_profiler


def demo_offline_generation(audio_path: str, output_path: str):
    """
    demo: generate music visualization offline (not real-time)
    full pipeline with audio analysis and prompt interpolation
    """
    print("=== hambaMetal offline generation demo ===\n")

    # use balanced preset (good quality, reasonable speed)
    config = PipelineConfig.balanced_preset()

    # override for faster demo
    config.inference.batch_size = 8
    config.inference.height = 384
    config.inference.width = 384

    print(f"config: {config.inference.height}x{config.inference.width}, batch={config.inference.batch_size}")

    # initialize components
    print("\n1. initializing audio analyzer...")
    audio_analyzer = AudioAnalyzer(config.audio)

    print("2. analyzing audio file...")
    audio_features = audio_analyzer.analyze(audio_path)
    print(f"   - duration: {audio_features.duration_seconds:.1f}s")
    print(f"   - tempo: {audio_features.tempo:.1f} bpm")
    print(f"   - frames: {audio_features.num_frames}")

    print("\n3. initializing diffusion pipeline...")
    base_pipeline = SD15Pipeline(
        diffusion_config=config.diffusion,
        inference_config=config.inference,
    )

    pipeline = StreamDiffusionPipeline(
        base_pipeline=base_pipeline,
        stream_config=config.stream,
    )

    print("4. initializing latent generator...")
    latent_gen = LatentGenerator(
        config=config.latent,
        device=config.diffusion.device,
        dtype=config.diffusion.get_torch_dtype(),
    )

    print("5. generating latent walk...")
    latent_shape = pipeline.get_latent_shape()
    latents = latent_gen.generate_latent_walk(
        audio_features=audio_features,
        latent_shape=latent_shape,
        precompute=True,  # generate all at once
    )
    print(f"   - latents shape: {latents.shape}")

    print("\n6. initializing prompt interpolator...")
    prompt_interp = PromptInterpolator(
        config=config.prompt,
        device=config.diffusion.device,
        dtype=config.diffusion.get_torch_dtype(),
    )

    print("7. encoding prompts...")
    profiler = get_profiler()

    with profiler.profile("encode_prompts"):
        base_embeds = pipeline.encode_prompt(config.prompt.base_prompt)
        chroma_embeds = pipeline.encode_prompt(config.prompt.chroma_prompts)

        # remove batch dim
        if base_embeds.dim() == 3:
            base_embeds = base_embeds[0]

    print("8. interpolating prompts based on music...")
    with profiler.profile("interpolate_prompts"):
        prompt_embeds = prompt_interp.interpolate_prompts(
            audio_features=audio_features,
            base_prompt_embeds=base_embeds,
            chroma_prompt_embeds=chroma_embeds,
        )
    print(f"   - prompt embeds shape: {prompt_embeds.shape}")

    print("\n9. generating frames...")
    with profiler.profile("generate_all_frames"):
        frames = pipeline.base.generate_batch(
            latents=latents,
            prompt_embeds=prompt_embeds,
        )
    print(f"   - generated {len(frames)} frames")

    print("\n10. profiling summary:")
    profiler.print_summary()

    print(f"\n=== generation complete ===")
    print(f"output would be saved to: {output_path}")
    print("(video export not implemented in demo)")


def demo_realtime_generation():
    """
    demo: real-time generation without audio
    simulates streaming use case
    """
    print("=== hambaMetal real-time generation demo ===\n")

    # fast preset for real-time
    config = PipelineConfig.fast_preset()

    # further optimize for demo
    config.inference.height = 256
    config.inference.width = 256
    config.stream.enable_temporal_coherence = True
    config.stream.temporal_blend_ratio = 0.4

    print(f"config: {config.inference.height}x{config.inference.width}, temporal coherence enabled")

    print("\n1. initializing pipeline...")
    base_pipeline = SD15Pipeline(
        diffusion_config=config.diffusion,
        inference_config=config.inference,
    )

    pipeline = StreamDiffusionPipeline(
        base_pipeline=base_pipeline,
        stream_config=config.stream,
    )

    print("2. initializing latent generator...")
    latent_gen = LatentGenerator(
        config=config.latent,
        device=config.diffusion.device,
        dtype=config.diffusion.get_torch_dtype(),
    )

    print("3. encoding prompts...")
    base_embeds = pipeline.encode_prompt(config.prompt.base_prompt)
    if base_embeds.dim() == 3:
        base_embeds = base_embeds[0:1]

    print("\n4. generating frames in real-time...")
    profiler = get_profiler()

    # simulate real-time with simple circular motion
    latent_shape = pipeline.get_latent_shape()

    torch.manual_seed(42)
    base_noise_x = torch.randn(latent_shape, device=config.diffusion.device, dtype=config.diffusion.get_torch_dtype())
    torch.manual_seed(43)
    base_noise_y = torch.randn(latent_shape, device=config.diffusion.device, dtype=config.diffusion.get_torch_dtype())

    num_frames = 20
    for i in range(num_frames):
        angle = (i / num_frames) * 2 * torch.pi * 2
        energy = 0.5 + 0.5 * torch.sin(angle * 2).item()

        # generate latent on-the-fly
        latent = torch.cos(angle) * base_noise_x + torch.sin(angle) * base_noise_y

        # generate frame with temporal coherence
        with profiler.profile("generate_frame"):
            frame = pipeline.generate_frame_with_coherence(
                latent=latent,
                prompt_embeds=base_embeds,
                music_energy=energy,
            )

        if (i + 1) % 5 == 0:
            avg_time = profiler.get_avg_time("generate_frame")
            print(f"   - frame {i+1}/{num_frames} | avg time: {avg_time:.1f}ms")

    print("\n5. profiling summary:")
    profiler.print_summary()

    print("\n=== real-time demo complete ===")


if __name__ == "__main__":
    print("hambaMetal - refactored music visualizer\n")
    print("choose demo mode:")
    print("1. offline generation (requires audio file)")
    print("2. real-time generation (no audio, simple motion)")

    choice = input("\nenter choice (1 or 2): ").strip()

    if choice == "1":
        audio_path = input("path to audio file: ").strip()
        output_path = input("output path (e.g., output.mp4): ").strip()

        if not Path(audio_path).exists():
            print(f"error: audio file not found: {audio_path}")
        else:
            demo_offline_generation(audio_path, output_path)

    elif choice == "2":
        demo_realtime_generation()

    else:
        print("invalid choice")
