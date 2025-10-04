"""
streamdiffusion-inspired pipeline with temporal coherence
key optimizations for smooth real-time generation
"""

from typing import Optional, Union
import torch

from .base import BaseDiffusionPipeline
from ..config import StreamConfig


class StreamDiffusionPipeline:
    """
    wrapper around any diffusion pipeline adding streamdiffusion optimizations:
    - temporal coherence (blend with previous frame)
    - latent buffer management
    - stochastic similarity filter (music-reactive)

    designed to wrap SD15Pipeline, SDXLPipeline, etc.
    """

    def __init__(
        self,
        base_pipeline: BaseDiffusionPipeline,
        stream_config: Optional[StreamConfig] = None,
    ):
        self.base = base_pipeline
        self.config = stream_config or StreamConfig()

        # temporal coherence state
        self.prev_latent: Optional[torch.Tensor] = None
        self.prev_prompt_embed: Optional[torch.Tensor] = None
        self.prev_pooled_embed: Optional[torch.Tensor] = None

        # similarity filter state
        self.prev_frame: Optional[torch.Tensor] = None
        self.skip_count: int = 0
        self.cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        # latent buffer for batch denoising (multi-step only)
        self.x_t_latent_buffer: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        """reset temporal state (call at start of new song)"""
        self.prev_latent = None
        self.prev_prompt_embed = None
        self.prev_pooled_embed = None
        self.prev_frame = None
        self.skip_count = 0
        self.x_t_latent_buffer = None

    @torch.no_grad()
    def generate_frame_with_coherence(
        self,
        latent: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        music_energy: float = 1.0,
    ) -> torch.Tensor:
        """
        generate single frame with temporal coherence
        blends current latent/prompt with previous for smooth transitions

        args:
            latent: target latent for this frame
            prompt_embeds: target prompt embedding
            pooled_prompt_embeds: pooled embeds (sdxl only)
            music_energy: current music energy [0, 1] for adaptive blending

        returns:
            generated frame tensor (C, H, W)
        """
        # apply temporal coherence if enabled
        if self.config.enable_temporal_coherence and self.prev_latent is not None:
            latent = self._blend_latent(latent, music_energy)
            prompt_embeds = self._blend_prompt(prompt_embeds)

            if pooled_prompt_embeds is not None and self.prev_pooled_embed is not None:
                pooled_prompt_embeds = self._blend_pooled_prompt(pooled_prompt_embeds)

        # check similarity filter (skip if too similar and low energy)
        if self.config.enable_similarity_filter:
            should_skip = self._should_skip_frame(latent, music_energy)
            if should_skip and self.prev_frame is not None:
                self.skip_count += 1
                return self.prev_frame.clone()

        # reset skip count
        self.skip_count = 0

        # generate frame
        frame = self.base.generate_single_frame(
            latent=latent,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )

        # store for next frame
        self.prev_latent = latent.clone()
        self.prev_prompt_embed = prompt_embeds.clone()
        if pooled_prompt_embeds is not None:
            self.prev_pooled_embed = pooled_prompt_embeds.clone()
        self.prev_frame = frame.clone()

        return frame

    def _blend_latent(self, target_latent: torch.Tensor, music_energy: float) -> torch.Tensor:
        """
        blend current latent with previous for smooth transitions
        blend ratio adapts to music energy (high energy = less blending)
        """
        if self.prev_latent is None:
            return target_latent

        # adaptive blend ratio based on music energy
        # high energy (loud music) -> use more of target (less blending)
        # low energy (quiet music) -> use more of previous (more blending)
        base_ratio = self.config.temporal_blend_ratio
        energy_factor = 1.0 - (music_energy * 0.5)  # scale energy influence
        blend_ratio = base_ratio * energy_factor

        # ensure ratio in valid range
        blend_ratio = max(0.0, min(0.8, blend_ratio))

        # blend: (1 - ratio) * target + ratio * previous
        blended = (1.0 - blend_ratio) * target_latent + blend_ratio * self.prev_latent

        return blended

    def _blend_prompt(self, target_prompt: torch.Tensor) -> torch.Tensor:
        """
        slerp between previous and current prompt embeddings
        smoother than linear interpolation for embedding space
        """
        if self.prev_prompt_embed is None:
            return target_prompt

        # use fixed blend ratio for prompts (less aggressive than latents)
        alpha = self.config.temporal_blend_ratio * 0.5

        # slerp interpolation
        return self._slerp(self.prev_prompt_embed, target_prompt, alpha)

    def _blend_pooled_prompt(self, target_pooled: torch.Tensor) -> torch.Tensor:
        """blend pooled embeddings (sdxl)"""
        if self.prev_pooled_embed is None:
            return target_pooled

        alpha = self.config.temporal_blend_ratio * 0.5
        return self._slerp(self.prev_pooled_embed, target_pooled, alpha)

    def _slerp(
        self,
        v0: torch.Tensor,
        v1: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        spherical linear interpolation between two tensors
        better than linear for high-dimensional embeddings
        """
        # normalize
        v0_norm = torch.nn.functional.normalize(v0, dim=-1)
        v1_norm = torch.nn.functional.normalize(v1, dim=-1)

        # compute angle
        dot = torch.sum(v0_norm * v1_norm, dim=-1, keepdim=True)
        omega = torch.acos(torch.clamp(dot, -1.0, 1.0))
        sin_omega = torch.sin(omega)

        # handle parallel vectors (fall back to lerp)
        if torch.any(sin_omega < 1e-6):
            return (1.0 - alpha) * v0 + alpha * v1

        # slerp
        coef_0 = torch.sin((1.0 - alpha) * omega) / sin_omega
        coef_1 = torch.sin(alpha * omega) / sin_omega

        return coef_0 * v0 + coef_1 * v1

    def _should_skip_frame(self, latent: torch.Tensor, music_energy: float) -> bool:
        """
        stochastic similarity filter - skip if too similar to previous
        adapted from streamdiffusion with music reactivity

        high energy music -> skip less (always generate new frames)
        low energy music -> skip more (reuse similar frames)
        """
        # don't skip if we've skipped too many already
        if self.skip_count >= self.config.max_skip_frames:
            return False

        # don't skip if no previous frame
        if self.prev_frame is None:
            return False

        # compute similarity between current and previous latent
        similarity = self.cos_sim(
            latent.reshape(-1),
            self.prev_latent.reshape(-1) if self.prev_latent is not None else latent.reshape(-1),
        ).item()

        # adaptive threshold based on music energy
        # quiet music (low energy) -> higher threshold -> more skipping
        # loud music (high energy) -> lower threshold -> less skipping
        base_threshold = self.config.similarity_threshold
        energy_adjustment = (1.0 - music_energy) * 0.04
        adaptive_threshold = min(0.99, base_threshold + energy_adjustment)

        # if below threshold, don't skip
        if similarity < adaptive_threshold:
            return False

        # probabilistic skip (streamdiffusion's key insight)
        # avoids hard cutoff that causes stuttering
        skip_prob = (similarity - adaptive_threshold) / (1.0 - adaptive_threshold)

        import random
        return random.random() < skip_prob

    # delegate other methods to base pipeline
    def encode_prompt(self, *args, **kwargs):
        """delegate to base pipeline"""
        return self.base.encode_prompt(*args, **kwargs)

    def supports_pooled_embeds(self) -> bool:
        """delegate to base pipeline"""
        return self.base.supports_pooled_embeds()

    def get_latent_shape(self):
        """delegate to base pipeline"""
        return self.base.get_latent_shape()

    def cleanup(self):
        """delegate to base pipeline"""
        return self.base.cleanup()
