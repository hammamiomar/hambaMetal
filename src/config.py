"""
type-safe configuration using pydantic for all pipeline components
"""

from typing import Literal, Optional, List
from pydantic import BaseModel, Field, field_validator
import torch


class DiffusionConfig(BaseModel):
    """core diffusion model configuration"""

    model_id: str = Field(
        default="runwayml/stable-diffusion-v1-5",
        description="huggingface model id or local path",
    )
    lora_id: Optional[str] = Field(
        default="ByteDance/Hyper-SD",
        description="lora weights to apply (None to skip)",
    )
    lora_filename: Optional[str] = Field(
        default="Hyper-SD15-1step-lora.safetensors",
        description="specific lora file to load",
    )
    device: Literal["cuda", "mps", "cpu"] = Field(
        default="mps", description="compute device"
    )
    torch_dtype: Literal["float16", "float32", "bfloat16"] = Field(
        default="float16", description="model precision"
    )
    use_tiny_vae: bool = Field(
        default=True, description="use tiny vae for 3-5x decode speedup"
    )
    tiny_vae_id: str = Field(
        default="madebyollin/taesd", description="tiny vae model id"
    )
    compile_unet: bool = Field(
        default=False, description="compile unet with torch.compile (not supported on mps)"
    )
    enable_attention_slicing: bool = Field(
        default=False, description="reduce vram usage at cost of speed"
    )
    enable_channels_last: bool = Field(
        default=False, description="use channels last memory format (cuda only)"
    )

    class Config:
        arbitrary_types_allowed = True  # allow torch.dtype

    @field_validator("torch_dtype", mode="before")
    @classmethod
    def validate_dtype(cls, v: str | torch.dtype) -> str:
        """convert torch dtype to string for pydantic"""
        if isinstance(v, torch.dtype):
            return str(v).split(".")[-1]
        return v

    def get_torch_dtype(self) -> torch.dtype:
        """get actual torch dtype from string"""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map[self.torch_dtype]


class StreamConfig(BaseModel):
    """streamdiffusion-inspired optimizations"""

    enable_temporal_coherence: bool = Field(
        default=True, description="blend with previous frame for smooth transitions"
    )
    temporal_blend_ratio: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="how much to blend with previous frame (0=none, 1=full)",
    )
    enable_batch_denoising: bool = Field(
        default=False,
        description="batch multiple denoising steps (requires multi-step inference)",
    )
    denoising_steps: int = Field(
        default=1,
        ge=1,
        le=10,
        description="number of denoising steps (1 for hyper-sd)",
    )
    frame_buffer_size: int = Field(
        default=1, ge=1, description="number of frames to buffer for batch denoising"
    )
    enable_similarity_filter: bool = Field(
        default=False,
        description="skip similar frames during quiet music sections",
    )
    similarity_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="cosine similarity threshold for skipping frames",
    )
    max_skip_frames: int = Field(
        default=10, ge=1, description="max consecutive frames to skip"
    )


class InferenceConfig(BaseModel):
    """inference-time parameters"""

    num_inference_steps: int = Field(default=1, ge=1, description="denoising steps")
    guidance_scale: float = Field(
        default=0.0, ge=0.0, description="cfg scale (0=no cfg)"
    )
    batch_size: int = Field(default=16, ge=1, description="frames per batch")
    height: int = Field(default=512, ge=64, description="output height")
    width: int = Field(default=512, ge=64, description="output width")
    eta: float = Field(
        default=1.0, ge=0.0, le=1.0, description="noise schedule parameter"
    )
    seed: int = Field(default=42069, description="random seed for reproducibility")

    @field_validator("height", "width")
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        """ensure dimensions are multiples of 8 for vae"""
        if v % 8 != 0:
            raise ValueError(f"dimension must be multiple of 8, got {v}")
        return v


class AudioConfig(BaseModel):
    """audio analysis parameters"""

    hop_length: int = Field(
        default=512, ge=64, description="hop length for stft analysis"
    )
    sr: int = Field(default=22050, ge=8000, description="target sample rate")
    n_mels: int = Field(default=256, ge=32, description="number of mel bands")
    n_chroma: int = Field(default=12, description="number of chroma bins")
    max_duration_minutes: float = Field(
        default=6.0, gt=0, description="max audio duration to prevent oom"
    )
    parallel_extraction: bool = Field(
        default=True, description="extract features in parallel"
    )


class LatentConfig(BaseModel):
    """latent walk generation parameters"""

    distance: float = Field(
        default=1.0, ge=0.0, description="radius of circular movement"
    )
    note_type: Literal["quarter", "half", "whole"] = Field(
        default="quarter", description="beat subdivision type"
    )
    use_onset_emphasis: bool = Field(
        default=True, description="emphasize latents at musical onsets"
    )
    onset_noise_scale: float = Field(
        default=0.3, ge=0.0, description="strength of onset emphasis noise"
    )
    enable_peak_oscillation: bool = Field(
        default=True, description="add oscillation at energy peaks"
    )
    oscillation_width: int = Field(
        default=5, ge=1, description="frames to spread oscillation"
    )


class PromptConfig(BaseModel):
    """prompt interpolation parameters"""

    interpolation_mode: Literal["clip", "slerp_sum", "onset_focus", "cumulative"] = (
        Field(default="cumulative", description="method for chroma-based interpolation")
    )
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="strength of chroma influence on prompts",
    )
    smoothing_factor: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="temporal smoothing to avoid jitter",
    )
    power_scale: float = Field(
        default=2.0, ge=1.0, description="exponent to emphasize strong chroma"
    )
    sigma_time: float = Field(
        default=2.0, ge=0.0, description="temporal gaussian smoothing sigma"
    )
    sigma_chroma: float = Field(
        default=1.0, ge=0.0, description="chroma gaussian smoothing sigma"
    )
    num_chromas_focus: int = Field(
        default=6, ge=1, le=12, description="number of chroma dims to consider"
    )
    num_prompt_shuffles: int = Field(
        default=4, ge=0, description="times to shuffle prompts for variety"
    )
    base_prompt: str = Field(
        default="abstract colorful flowing shapes",
        description="base prompt for all frames",
    )
    chroma_prompts: List[str] = Field(
        default_factory=lambda: [
            "red warm energetic",
            "orange vibrant dynamic",
            "yellow bright cheerful",
            "green natural calm",
            "cyan cool flowing",
            "blue deep peaceful",
            "purple mysterious dreamy",
            "magenta electric intense",
            "pink soft gentle",
            "brown earthy grounded",
            "white pure luminous",
            "black dark dramatic",
        ],
        description="12 prompts corresponding to chromatic scale (C, C#, D, ...)",
    )

    @field_validator("chroma_prompts")
    @classmethod
    def validate_chroma_count(cls, v: List[str]) -> List[str]:
        """ensure exactly 12 chroma prompts"""
        if len(v) != 12:
            raise ValueError(f"must provide exactly 12 chroma prompts, got {len(v)}")
        return v


class ProfilingConfig(BaseModel):
    """performance profiling settings"""

    enable_profiling: bool = Field(default=True, description="enable mps profiling")
    log_interval: int = Field(
        default=10, ge=1, description="log metrics every n frames"
    )
    track_memory: bool = Field(default=True, description="track memory usage")
    export_json: bool = Field(
        default=True, description="export profiling results to json"
    )
    json_path: str = Field(
        default="profiling_results.json", description="path for json export"
    )


class PipelineConfig(BaseModel):
    """complete pipeline configuration bundling all sub-configs"""

    diffusion: DiffusionConfig = Field(default_factory=DiffusionConfig)
    stream: StreamConfig = Field(default_factory=StreamConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    latent: LatentConfig = Field(default_factory=LatentConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)

    @classmethod
    def fast_preset(cls) -> "PipelineConfig":
        """optimized for speed (lower quality)"""
        return cls(
            diffusion=DiffusionConfig(use_tiny_vae=True),
            stream=StreamConfig(
                enable_temporal_coherence=True,
                temporal_blend_ratio=0.4,
                enable_similarity_filter=True,
            ),
            inference=InferenceConfig(
                batch_size=32, height=384, width=384, num_inference_steps=1
            ),
        )

    @classmethod
    def balanced_preset(cls) -> "PipelineConfig":
        """balanced speed and quality"""
        return cls(
            diffusion=DiffusionConfig(use_tiny_vae=True),
            stream=StreamConfig(
                enable_temporal_coherence=True, temporal_blend_ratio=0.3
            ),
            inference=InferenceConfig(
                batch_size=16, height=512, width=512, num_inference_steps=1
            ),
        )

    @classmethod
    def quality_preset(cls) -> "PipelineConfig":
        """optimized for quality (slower)"""
        return cls(
            diffusion=DiffusionConfig(use_tiny_vae=False),
            stream=StreamConfig(
                enable_temporal_coherence=True,
                temporal_blend_ratio=0.2,
                enable_batch_denoising=True,
                denoising_steps=2,
            ),
            inference=InferenceConfig(
                batch_size=8, height=512, width=512, num_inference_steps=2
            ),
        )
