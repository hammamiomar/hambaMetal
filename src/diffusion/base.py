"""
abstract base pipeline for diffusion models
flexible interface supporting sd1.5, sdxl, flux, etc.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, Tuple

import torch
from diffusers import (
    StableDiffusionPipeline,
    AutoencoderTiny,
    DiffusionPipeline,
)
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import hf_hub_download

from ..config import DiffusionConfig, InferenceConfig


class BaseDiffusionPipeline(ABC):
    """
    abstract base for all diffusion pipelines
    defines standard interface for music visualization
    """

    def __init__(
        self,
        diffusion_config: DiffusionConfig,
        inference_config: InferenceConfig,
    ):
        self.diffusion_config = diffusion_config
        self.inference_config = inference_config

        # pipeline components (initialized by subclass)
        self.pipe: Optional[DiffusionPipeline] = None
        self.text_encoder = None
        self.unet = None
        self.vae = None
        self.scheduler = None
        self.image_processor: Optional[VaeImageProcessor] = None

        # device and dtype
        self.device = torch.device(diffusion_config.device)
        self.dtype = diffusion_config.get_torch_dtype()

        # random seed for reproducibility
        torch.manual_seed(inference_config.seed)
        if diffusion_config.device == "cuda":
            torch.cuda.manual_seed_all(inference_config.seed)

        # performance tracking
        self.inference_time_ema: float = 0.0

    @abstractmethod
    def load_model(self) -> None:
        """
        load and configure the diffusion model
        implemented by subclasses for specific model types
        """
        pass

    @abstractmethod
    def encode_prompt(
        self,
        prompt: Union[str, list[str]],
        negative_prompt: Optional[Union[str, list[str]]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        encode text prompts to embeddings
        returns different formats for sd1.5 vs sdxl (pooled embeds)
        """
        pass

    @abstractmethod
    def supports_pooled_embeds(self) -> bool:
        """check if model uses pooled embeddings (sdxl)"""
        pass

    def _apply_optimizations(self) -> None:
        """
        apply performance optimizations based on config
        called after model loading
        """
        if self.pipe is None:
            raise RuntimeError("must load model before applying optimizations")

        # tiny vae replacement (3-5x decode speedup)
        if self.diffusion_config.use_tiny_vae:
            print("replacing vae with tiny vae for 3-5x speedup...")
            self.vae = AutoencoderTiny.from_pretrained(
                self.diffusion_config.tiny_vae_id,
                torch_dtype=self.dtype,
            ).to(self.device)
            self.pipe.vae = self.vae

        # channels last memory format (cuda only, better tensor core usage)
        if (
            self.diffusion_config.enable_channels_last
            and self.diffusion_config.device == "cuda"
        ):
            print("enabling channels last memory format...")
            self.unet = self.unet.to(memory_format=torch.channels_last)
            self.vae = self.vae.to(memory_format=torch.channels_last)

        # attention slicing (reduce vram at cost of speed)
        if self.diffusion_config.enable_attention_slicing:
            print("enabling attention slicing...")
            self.pipe.enable_attention_slicing()

        # torch compile (cuda only, not supported on mps)
        if (
            self.diffusion_config.compile_unet
            and hasattr(torch, "compile")
            and self.diffusion_config.device == "cuda"
        ):
            print("compiling unet with torch.compile...")
            try:
                self.unet = torch.compile(
                    self.unet,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                print("unet compilation successful")
            except Exception as e:
                print(f"unet compilation failed: {e}")

        # disable safety checker and progress bars
        if hasattr(self.pipe, "safety_checker"):
            self.pipe.safety_checker = None
        if hasattr(self.pipe, "set_progress_bar_config"):
            self.pipe.set_progress_bar_config(disable=True)

    def _load_lora_weights(
        self,
        lora_id: Optional[str] = None,
        lora_filename: Optional[str] = None,
    ) -> None:
        """
        load and fuse lora weights
        used for hyper-sd, lcm, etc.
        """
        if lora_id is None:
            return

        print(f"loading lora weights from {lora_id}...")

        if lora_filename:
            # download specific file
            lora_path = hf_hub_download(lora_id, lora_filename)
            self.pipe.load_lora_weights(lora_path)
        else:
            # load from repo
            self.pipe.load_lora_weights(lora_id)

        # fuse lora into model weights for faster inference
        print("fusing lora weights...")
        self.pipe.fuse_lora()

    @torch.no_grad()
    def generate_single_frame(
        self,
        latent: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        generate single frame from latent and prompt embeddings
        returns decoded image tensor (C, H, W) in range [0, 1]
        """
        # ensure inputs are on correct device
        latent = latent.to(device=self.device, dtype=self.dtype)
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dtype)

        # ensure latent has batch dimension
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)

        # prepare kwargs
        kwargs = {
            "prompt_embeds": prompt_embeds,
            "latents": latent,
            "num_inference_steps": self.inference_config.num_inference_steps,
            "guidance_scale": self.inference_config.guidance_scale,
            "eta": self.inference_config.eta,
            "output_type": "pt",  # return pytorch tensor
        }

        # add pooled embeds for sdxl
        if pooled_prompt_embeds is not None:
            if pooled_prompt_embeds.dim() == 1:
                pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(0)
            kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds.to(
                device=self.device, dtype=self.dtype
            )

        # generate
        output = self.pipe(**kwargs)

        # extract image tensor (B, C, H, W)
        image = output.images[0]  # remove batch dim

        return image

    @torch.no_grad()
    def generate_batch(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        """
        generate batch of frames from latents and prompt embeddings
        batches are processed according to inference_config.batch_size
        returns list of image tensors
        """
        num_frames = len(latents)
        batch_size = self.inference_config.batch_size
        frames = []

        for i in range(0, num_frames, batch_size):
            batch_end = min(i + batch_size, num_frames)

            # prepare batch
            latent_batch = latents[i:batch_end]
            prompt_batch = prompt_embeds[i:batch_end]

            # prepare kwargs
            kwargs = {
                "prompt_embeds": prompt_batch,
                "latents": latent_batch,
                "num_inference_steps": self.inference_config.num_inference_steps,
                "guidance_scale": self.inference_config.guidance_scale,
                "eta": self.inference_config.eta,
                "output_type": "pt",
            }

            # add pooled embeds for sdxl
            if pooled_prompt_embeds is not None:
                kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds[i:batch_end]

            # generate batch
            output = self.pipe(**kwargs)
            frames.extend(output.images)

        return frames

    def get_latent_shape(self) -> Tuple[int, int, int, int]:
        """
        get shape for latent tensors (B, C, H, W)
        based on model's vae scale factor
        """
        vae_scale_factor = getattr(
            self.pipe, "vae_scale_factor", 8
        )  # default to 8 for sd models

        latent_h = self.inference_config.height // vae_scale_factor
        latent_w = self.inference_config.width // vae_scale_factor
        latent_c = self.unet.config.in_channels

        return (1, latent_c, latent_h, latent_w)

    def cleanup(self) -> None:
        """cleanup gpu memory"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()
