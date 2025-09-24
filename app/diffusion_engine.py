import asyncio
import base64
import logging
import os
import sys
import time
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.profiler
from config import Config
from diffusers import AutoencoderTiny, DiffusionPipeline, TCDScheduler
from huggingface_hub import hf_hub_download
from PIL import Image

# Add the parent directories to the path to import StreamDiffusion
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from streamdiffusion import StreamDiffusion

logger = logging.getLogger("uvicorn")


class ProfilerPerformanceTracker:
    """
    Professional performance tracking using PyTorch profiler.
    """
    def __init__(self, window_size: int = 10, enable_profiling: bool = True):
        self.window_size = window_size
        self.total_generations = 0
        self.enable_profiling = enable_profiling

        # Simple timing data for basic metrics
        self.total_times = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.peak_memory = 0.0

        # Profiler data
        self.profiler_results = deque(maxlen=5)  # Keep last 5 profiling sessions

    def create_profiler(self) -> Optional[torch.profiler.profile]:
        """Create a PyTorch profiler instance."""
        if not self.enable_profiling:
            return None

        # Use only CPU profiling for MPS - CUDA profiling not supported
        activities = [torch.profiler.ProfilerActivity.CPU]

        return torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )

    def _analyze_profiler_output(self, key_averages) -> Dict[str, Any]:
        """Analyze PyTorch profiler output to extract insights."""
        analysis = {
            "total_ops": len(key_averages),
            "top_operations": [],
            "memory_intensive_ops": [],
            "bottlenecks": []
        }

        # Sort by self CPU time
        sorted_ops = sorted(key_averages, key=lambda x: x.self_cpu_time_total, reverse=True)

        # Get top 5 time-consuming operations
        for i, op in enumerate(sorted_ops[:5]):
            analysis["top_operations"].append({
                "name": op.key,
                "cpu_time_total": op.cpu_time_total,
                "self_cpu_time_total": op.self_cpu_time_total,
                "cpu_time_avg": op.cpu_time_total / max(op.count, 1),
                "count": op.count
            })

        # Find memory-intensive operations
        for op in sorted_ops:
            if hasattr(op, 'cpu_memory_usage') and op.cpu_memory_usage > 1024 * 1024:  # > 1MB
                analysis["memory_intensive_ops"].append({
                    "name": op.key,
                    "memory_usage": op.cpu_memory_usage,
                    "count": op.count
                })

        return analysis

    def add_measurement(self, timing_data: Dict[str, float]):
        """Add basic timing measurement."""
        self.total_times.append(timing_data.get('total_time', 0))

        if 'memory_usage' in timing_data:
            self.memory_usage.append(timing_data['memory_usage'])
            self.peak_memory = max(self.peak_memory, timing_data['memory_usage'])

        self.total_generations += 1

    def get_current_fps(self) -> float:
        """Get FPS from the most recent measurement."""
        if not self.total_times:
            return 0.0
        return 1.0 / self.total_times[-1]

    def get_average_fps(self) -> float:
        """Get average FPS over the rolling window."""
        if not self.total_times:
            return 0.0
        avg_time = sum(self.total_times) / len(self.total_times)
        return 1.0 / avg_time

    def get_min_fps(self) -> float:
        """Get minimum FPS (maximum time)."""
        if not self.total_times:
            return 0.0
        max_time = max(self.total_times)
        return 1.0 / max_time

    def get_max_fps(self) -> float:
        """Get maximum FPS (minimum time)."""
        if not self.total_times:
            return 0.0
        min_time = min(self.total_times)
        return 1.0 / min_time

    def get_profiler_insights(self) -> Dict[str, Any]:
        """Get insights from PyTorch profiler."""
        if not self.profiler_results:
            return {"status": "No profiling data available"}

        latest_results = self.profiler_results[-1]

        insights = {
            "status": "active",
            "total_operations": latest_results.get("total_ops", 0),
            "top_bottlenecks": [],
            "optimization_suggestions": []
        }

        # Extract top bottlenecks
        for op in latest_results.get("top_operations", [])[:3]:
            insights["top_bottlenecks"].append({
                "operation": op["name"],
                "avg_time_ms": op["cpu_time_avg"] / 1000,  # Convert to ms
                "total_calls": op["count"]
            })

        # Generate optimization suggestions
        if latest_results.get("memory_intensive_ops"):
            insights["optimization_suggestions"].append(
                "Memory-intensive operations detected. Consider gradient checkpointing or mixed precision."
            )

        return insights

    def get_timing_breakdown(self) -> Dict[str, Any]:
        """Get simplified timing breakdown for compatibility."""
        if not self.profiler_results:
            return {}

        latest_results = self.profiler_results[-1]

        # Convert profiler results to timing breakdown format
        breakdown = {
            "total_operations": latest_results.get("total_ops", 0),
            "top_operations": latest_results.get("top_operations", []),
            "bottleneck": "PyTorch Profiler Active"
        }

        return breakdown

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if not self.memory_usage:
            return {}

        current_memory = self.memory_usage[-1] if self.memory_usage else 0
        avg_memory = sum(self.memory_usage) / len(self.memory_usage)

        return {
            "current_memory_mb": current_memory,
            "average_memory_mb": avg_memory,
            "peak_memory_mb": self.peak_memory,
            "memory_efficiency": (avg_memory / self.peak_memory * 100) if self.peak_memory > 0 else 0
        }

    def export_trace(self, profiler: torch.profiler.profile, filepath: str = None) -> Optional[str]:
        """Export profiler trace for analysis."""
        if not profiler:
            return None

        if filepath is None:
            filepath = f"./profiler_trace_{int(time.time())}.json"

        try:
            profiler.export_chrome_trace(filepath)
            return filepath
        except Exception as e:
            logger.error(f"Failed to export trace: {e}")
            return None


class DiffusionEngine:
    """
    StreamDiffusion engine handling all torch operations and model management.
    """
    def __init__(self, config: Config):
        self.config = config

        # Initialize performance tracking
        self.performance_tracker = ProfilerPerformanceTracker(
            window_size=config.fps_window_size,
            enable_profiling=config.enable_profiling
        )

        # Setup pipeline and stream
        self.pipe = None
        self.stream = None

        # Lock for thread safety
        self._generation_lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the diffusion pipeline and stream."""
        logger.info("Initializing DiffusionEngine...")

        # Setup pipeline and stream
        self.pipe = self._setup_pipeline()
        self.stream = self._setup_stream_diffusion()

        # Warmup the model
        await self._warmup()

        logger.info("DiffusionEngine initialization complete")

    def _setup_pipeline(self) -> DiffusionPipeline:
        """
        Setup pipeline with Hyper-SD LoRA and TinyVAE for MPS optimization.
        """
        logger.info("Loading base model...")
        pipe = DiffusionPipeline.from_pretrained(
            self.config.base_model_id,
            dtype=self.config.dtype,  # Fixed: torch_dtype -> dtype
            variant="fp16"
        ).to(self.config.device)

        logger.info("Loading Hyper-SD LoRA...")
        lora_path = hf_hub_download(
            self.config.hyper_sd_repo,
            self.config.hyper_sd_lora
        )
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora()

        logger.info("Setting up TCD scheduler...")
        pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

        # Replace with TinyVAE for faster encoding/decoding
        if self.config.use_tiny_vae:
            logger.info(f"Loading TinyVAE: {self.config.vae_id}")
            pipe.vae = AutoencoderTiny.from_pretrained(
                self.config.vae_id,
                dtype=self.config.dtype  # Fixed: torch_dtype -> dtype
            ).to(self.config.device)
            logger.info("TinyVAE loaded - 4x faster VAE operations enabled")

        if self.config.acceleration == "xformers":
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("XFormers acceleration enabled")
            except Exception as e:
                logger.warning(f"XFormers not available: {e}")

        logger.info(f"Pipeline optimized for {self.config.device}")
        return pipe

    def _setup_stream_diffusion(self) -> StreamDiffusion:
        """
        Setup StreamDiffusion with manual configuration.
        """
        logger.info("Setting up StreamDiffusion...")
        stream = StreamDiffusion(
            pipe=self.pipe,
            t_index_list=self.config.t_index_list,
            dtype=self.config.dtype,  # Fixed: torch_dtype -> dtype
            width=self.config.width,
            height=self.config.height,
            do_add_noise=True,
            use_denoising_batch=self.config.use_denoising_batch,
            frame_buffer_size=self.config.frame_buffer_size,
            cfg_type=self.config.cfg_type,
        )

        # Manual prepare call
        logger.info("Preparing StreamDiffusion...")

        # MPS-compatible generator setup - always use CPU generator for MPS compatibility
        generator = torch.Generator()
        generator.manual_seed(42)

        stream.prepare(
            prompt="",  # Will be updated dynamically
            negative_prompt="",
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            delta=1.0,
            generator=generator,
            seed=42,
        )

        return stream

    async def _warmup(self):
        """
        Warmup the model with several inference runs.
        """
        logger.info(f"Warming up with {self.config.warmup_iterations} iterations...")
        for i in range(self.config.warmup_iterations):
            start_time = time.perf_counter()
            _ = self.stream.txt2img(batch_size=1)
            end_time = time.perf_counter()
            logger.info(f"Warmup {i+1}/{self.config.warmup_iterations}: {(end_time - start_time)*1000:.2f}ms")

    def _get_memory_usage(self) -> float:
        """Get current MPS memory usage in MB."""
        if torch.backends.mps.is_available() and hasattr(torch.mps, 'current_allocated_memory'):
            return torch.mps.current_allocated_memory() / 1024 / 1024
        return 0.0

    async def generate_image(self, prompt: str) -> Dict[str, Any]:
        """
        Generate image with PyTorch profiler optimization.
        """
        async with self._generation_lock:
            # Memory tracking
            memory_start = self._get_memory_usage()
            total_start = time.perf_counter()

            # Update prompt if provided
            if prompt.strip():
                self.stream.update_prompt(prompt)

            # Generate image with optional profiling
            profiler_data = None

            try:
                if self.config.enable_profiling:
                    # Use profiler context manager directly
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU],
                        record_shapes=True,
                        profile_memory=True
                    ) as prof:
                        image_tensor = self.stream.txt2img(batch_size=1)

                    # Get profiler insights
                    try:
                        key_averages = prof.key_averages()
                        analysis = self.performance_tracker._analyze_profiler_output(key_averages)
                        profiler_data = {
                            "status": "active",
                            "total_operations": analysis.get("total_ops", 0),
                            "top_bottlenecks": [
                                {
                                    "operation": op["name"],
                                    "avg_time_ms": op["cpu_time_avg"] / 1000,
                                    "total_calls": op["count"]
                                }
                                for op in analysis.get("top_operations", [])[:3]
                            ]
                        }
                    except Exception as e:
                        logger.warning(f"Profiler analysis failed: {e}")
                        profiler_data = {"status": "analysis_failed", "error": str(e)}
                else:
                    # Standard generation without profiling
                    image_tensor = self.stream.txt2img(batch_size=1)

            except Exception as e:
                logger.error(f"Generation error: {e}")
                raise

            # Post-processing with minimal overhead
            websocket_start = time.perf_counter()

            # Convert to PIL and base64
            image = self._process_tensor_to_image(image_tensor)
            base64_image = self._pil_to_base64(image)
            websocket_time = time.perf_counter() - websocket_start

            # Calculate total time and memory
            total_time = time.perf_counter() - total_start
            memory_end = self._get_memory_usage()

            # Create simplified timing data for basic tracking
            timing_data = {
                'total_time': total_time,
                'memory_usage': max(memory_start, memory_end)
            }

            # Track performance
            self.performance_tracker.add_measurement(timing_data)
            current_fps = self.performance_tracker.get_current_fps()

            result = {
                "base64_image": base64_image,
                "fps": current_fps,
                "inference_time_ms": total_time * 1000,
                "stats": self.get_stats_dict()
            }

            # Add profiler insights if available
            if profiler_data:
                result["profiler_insights"] = profiler_data

            return result

    def _process_tensor_to_image(self, image_tensor) -> Image.Image:
        """Convert tensor to PIL Image with robust shape handling."""
        if isinstance(image_tensor, torch.Tensor):
            logger.debug(f"Image tensor shape: {image_tensor.shape}")

            # Handle different tensor shapes from StreamDiffusion
            if len(image_tensor.shape) == 4:  # (batch, channels, height, width)
                image_tensor = image_tensor.squeeze(0)  # Remove batch dimension
            elif len(image_tensor.shape) == 3 and image_tensor.shape[0] == 1:  # (1, height, width)
                image_tensor = image_tensor.squeeze(0)  # Remove first dimension

            # Ensure we have (height, width, channels) or (channels, height, width)
            if len(image_tensor.shape) == 3:
                if image_tensor.shape[0] == 3:  # (channels, height, width)
                    image_tensor = image_tensor.permute(1, 2, 0)  # Convert to (height, width, channels)
                elif image_tensor.shape[2] == 3:  # Already (height, width, channels)
                    pass  # Keep as is
                else:
                    logger.error(f"Unexpected tensor shape: {image_tensor.shape}")
                    # Try to handle single channel
                    if image_tensor.shape[0] == 1:
                        image_tensor = image_tensor.squeeze(0)  # Remove channel dim
                        image_tensor = image_tensor.unsqueeze(2).repeat(1, 1, 3)  # Convert to RGB
                    else:
                        raise ValueError(f"Cannot handle tensor shape: {image_tensor.shape}")
            elif len(image_tensor.shape) == 2:  # (height, width) - grayscale
                image_tensor = image_tensor.unsqueeze(2).repeat(1, 1, 3)  # Convert to RGB
            else:
                raise ValueError(f"Cannot handle tensor shape: {image_tensor.shape}")

            # Convert to numpy and ensure correct data type
            image_array = image_tensor.detach().cpu().numpy()

            # Normalize to 0-255 range if needed
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype('uint8')
            else:
                image_array = image_array.astype('uint8')

            # Ensure we have the right shape for PIL
            if image_array.shape[2] != 3:
                raise ValueError(f"Expected 3 channels, got {image_array.shape[2]}")

            return Image.fromarray(image_array)
        else:
            # Assume it's already a PIL Image
            return image_tensor

    def get_stats_dict(self) -> Dict[str, Any]:
        """
        Get comprehensive benchmark statistics as dictionary.
        """
        basic_stats = {
            "current_fps": self.performance_tracker.get_current_fps(),
            "average_fps": self.performance_tracker.get_average_fps(),
            "min_fps": self.performance_tracker.get_min_fps(),
            "max_fps": self.performance_tracker.get_max_fps(),
            "total_generations": self.performance_tracker.total_generations
        }

        if self.config.detailed_timing:
            basic_stats["timing_breakdown"] = self.performance_tracker.get_timing_breakdown()

        if self.config.memory_tracking:
            basic_stats["memory_stats"] = self.performance_tracker.get_memory_stats()

        return basic_stats

    def _pil_to_base64(self, image: Image.Image, format: str = "JPEG") -> str:
        """
        Convert a PIL image to base64.
        """
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode("ascii")

    async def benchmark_generator(self, prompt: str):
        """
        Async generator for benchmark image generation.
        """
        generation_count = 0
        while True:
            try:
                result = await self.generate_image(prompt)
                generation_count += 1
                yield result

                # Yield control to event loop every few generations
                if generation_count % 5 == 0:
                    await asyncio.sleep(0)

            except Exception as e:
                logger.error(f"Generation error in benchmark: {e}")
                break