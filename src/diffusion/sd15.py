"""
stable diffusion 1.5 pipeline implementation
supports hyper-sd and lcm loras
"""

from typing import Optional, Union

import torch
from diffusers import StableDiffusionPipeline, TCDScheduler, LCMScheduler
from diffusers.image_processor import VaeImageProcessor

from .base import BaseDiffusionPipeline
from ..config import DiffusionConfig, InferenceConfig


class SD15Pipeline(BaseDiffusionPipeline):
    """
    stable diffusion 1.5 pipeline
    default config uses hyper-sd for 1-step inference
    """

    def __init__(
        self,
        diffusion_config: Optional[DiffusionConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
    ):
        # use default configs if not provided
        diffusion_config = diffusion_config or DiffusionConfig()
        inference_config = inference_config or InferenceConfig()

        super().__init__(diffusion_config, inference_config)

        # load model
        self.load_model()

    def load_model(self) -> None:
        """load sd1.5 model with optimizations"""
        print(f"loading sd1.5 model: {self.diffusion_config.model_id}")

        # load base pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.diffusion_config.model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            safety_checker=None,
        )

        # move to device
        self.pipe = self.pipe.to(self.device)

        # extract components
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.pipe.vae_scale_factor
        )

        # load lora weights (hyper-sd, lcm, etc)
        self._load_lora_weights(
            lora_id=self.diffusion_config.lora_id,
            lora_filename=self.diffusion_config.lora_filename,
        )

        # configure scheduler for 1-step inference
        # tcd scheduler for hyper-sd, lcm scheduler for lcm
        if "hyper" in (self.diffusion_config.lora_id or "").lower():
            print("using tcd scheduler for hyper-sd")
            self.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)
        elif "lcm" in (self.diffusion_config.lora_id or "").lower():
            print("using lcm scheduler")
            self.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        self.pipe.scheduler = self.scheduler

        # apply optimizations (tiny vae, compile, etc)
        self._apply_optimizations()

        print("sd1.5 pipeline ready")

    def encode_prompt(
        self,
        prompt: Union[str, list[str]],
        negative_prompt: Optional[Union[str, list[str]]] = None,
    ) -> torch.Tensor:
        """
        encode text prompts to embeddings
        sd1.5 returns single embedding tensor (no pooled embeds)
        """
        # handle single prompt
        if isinstance(prompt, str):
            prompt = [prompt]

        # determine if we need cfg
        do_cfg = self.inference_config.guidance_scale > 1.0

        # encode
        prompt_embeds, negative_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompt if do_cfg else None,
        )

        # return just positive embeds if no cfg
        if not do_cfg:
            return prompt_embeds

        # concat for cfg
        return torch.cat([negative_embeds, prompt_embeds], dim=0)

    def supports_pooled_embeds(self) -> bool:
        """sd1.5 does not use pooled embeddings"""
        return False
