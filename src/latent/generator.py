"""
latent walk generation synchronized to music features
refactored from original getBeatLatentsCircle with type safety
"""

import math
from typing import Tuple, Literal, Optional
import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..audio.types import AudioFeatures
from ..config import LatentConfig


class LatentGenerator:
    """
    generate latent walks that move in response to music features
    supports circular motion, onset emphasis, and tempo adaptation
    """

    def __init__(
        self,
        config: Optional[LatentConfig] = None,
        device: str = "mps",
        dtype: torch.dtype = torch.float16,
    ):
        self.config = config or LatentConfig()
        self.device = torch.device(device)
        self.dtype = dtype

        # base noise vectors (initialized on first generation)
        self.base_noise_x: Optional[torch.Tensor] = None
        self.base_noise_y: Optional[torch.Tensor] = None
        self.onset_noise: Optional[torch.Tensor] = None

    def generate_latent_walk(
        self,
        audio_features: AudioFeatures,
        latent_shape: Tuple[int, int, int, int],
        precompute: bool = True,
    ) -> torch.Tensor:
        """
        generate latent walk for entire song

        args:
            audio_features: extracted audio features
            latent_shape: (batch, channels, height, width) for latents
            precompute: if True, generate all latents upfront
                       if False, only initialize base noise (use get_latent_at_frame)

        returns:
            latent tensor (num_frames, C, H, W) if precompute=True
            otherwise returns empty tensor (use get_latent_at_frame for on-the-fly)
        """
        # initialize base noise if not done
        if self.base_noise_x is None:
            self._initialize_base_noise(latent_shape)

        if not precompute:
            # on-the-fly mode: just return placeholder
            return torch.empty(0)

        # precompute mode: generate all latents
        num_frames = audio_features.num_frames
        all_latents = []

        # compute angles for circular motion
        angles = self._compute_circular_angles(audio_features)

        # compute onset emphasis
        onset_emphasis = self._compute_onset_emphasis(audio_features)

        # generate latents frame by frame
        for frame_idx in range(num_frames):
            latent = self._compute_latent_at_frame(
                frame_idx=frame_idx,
                angle=angles[frame_idx],
                onset_factor=onset_emphasis[frame_idx],
            )
            all_latents.append(latent)

        return torch.stack(all_latents, dim=0)

    def get_latent_at_frame(
        self,
        frame_idx: int,
        audio_features: AudioFeatures,
        latent_shape: Optional[Tuple[int, int, int, int]] = None,
    ) -> torch.Tensor:
        """
        generate latent for specific frame on-the-fly
        more memory efficient than precomputing all latents

        args:
            frame_idx: which frame to generate
            audio_features: audio features
            latent_shape: shape if base noise not initialized

        returns:
            latent tensor (1, C, H, W)
        """
        # initialize if needed
        if self.base_noise_x is None:
            if latent_shape is None:
                raise ValueError("must provide latent_shape for first call")
            self._initialize_base_noise(latent_shape)

        # compute angle and onset for this frame
        angles = self._compute_circular_angles(audio_features)
        onset_emphasis = self._compute_onset_emphasis(audio_features)

        angle = angles[frame_idx].item()
        onset_factor = onset_emphasis[frame_idx].item()

        return self._compute_latent_at_frame(frame_idx, angle, onset_factor)

    def _initialize_base_noise(self, latent_shape: Tuple[int, int, int, int]) -> None:
        """initialize base noise vectors for circular motion"""
        _, c, h, w = latent_shape

        # use fixed seeds for reproducibility
        torch.manual_seed(42)
        self.base_noise_x = torch.randn(
            (1, c, h, w), dtype=self.dtype, device=self.device
        )

        torch.manual_seed(43)
        self.base_noise_y = torch.randn(
            (1, c, h, w), dtype=self.dtype, device=self.device
        )

        if self.config.use_onset_emphasis:
            torch.manual_seed(44)
            self.onset_noise = torch.randn(
                (1, c, h, w), dtype=self.dtype, device=self.device
            )
        else:
            self.onset_noise = None

    def _compute_circular_angles(self, audio_features: AudioFeatures) -> torch.Tensor:
        """
        compute angles for circular latent walk
        synchronized to beats with tempo adaptation and energy scaling
        """
        num_frames = audio_features.num_frames

        # get beat frames based on note type
        beat_frames = self._get_beat_frames(audio_features)

        # tempo adaptation
        tempo_factor = min(1.5, max(0.5, audio_features.tempo / 120.0))
        dynamic_distance = self.config.distance * tempo_factor

        # energy values
        energy = torch.tensor(
            audio_features.mel_spec_normalized, dtype=self.dtype, device=self.device
        )

        # initialize angles
        angles = torch.zeros(num_frames, dtype=self.dtype, device=self.device)

        # compute angle changes between beats
        for i in range(len(beat_frames) - 1):
            start_frame = beat_frames[i]
            end_frame = beat_frames[i + 1]
            duration = end_frame - start_frame

            if duration <= 0:
                continue

            # energy at beat points
            current_energy = energy[start_frame]
            next_energy = energy[end_frame]

            # movement direction based on energy change
            energy_ratio = next_energy / (current_energy + 1e-5)
            direction = 1.0 if energy_ratio > 1.0 else -1.0

            # total angle change for this beat interval
            total_angle = (
                direction * next_energy.item() * dynamic_distance * 2 * math.pi
            )

            # apply easing
            t = torch.linspace(0, 1, steps=duration, device=self.device)
            eased_t = self._cubic_easing(t)

            # assign angles
            angles[start_frame:end_frame] = total_angle * eased_t

        # handle frames after last beat
        if beat_frames[-1] < num_frames - 1:
            start_frame = beat_frames[-1]
            end_frame = num_frames
            duration = end_frame - start_frame

            if duration > 0:
                fade_angle = energy[start_frame] * self.config.distance * 0.5 * math.pi
                t = torch.linspace(0, 1, steps=duration, device=self.device)
                fade_t = torch.pow(t, 1.5)
                angles[start_frame:end_frame] = fade_angle * fade_t

        # cumulative sum for continuous motion
        cumulative_angles = torch.cumsum(angles, dim=0)

        # add peak oscillations if enabled
        if self.config.enable_peak_oscillation:
            oscillation = self._compute_peak_oscillation(energy)
            cumulative_angles = cumulative_angles + oscillation

        return cumulative_angles

    def _compute_onset_emphasis(self, audio_features: AudioFeatures) -> torch.Tensor:
        """compute onset emphasis factor for each frame"""
        if not self.config.use_onset_emphasis:
            return torch.zeros(
                audio_features.num_frames, dtype=self.dtype, device=self.device
            )

        # create onset map
        onset_map = np.zeros(audio_features.num_frames)
        onset_map[audio_features.onset_frames] = 1.0

        # smooth with gaussian
        onset_map = gaussian_filter1d(onset_map, sigma=1.0)

        return torch.tensor(onset_map, dtype=self.dtype, device=self.device)

    def _compute_peak_oscillation(self, energy: torch.Tensor) -> torch.Tensor:
        """add oscillation at energy peaks"""
        num_frames = len(energy)

        # find peaks (80th percentile)
        sorted_energy, _ = torch.sort(energy)
        threshold_idx = int(0.8 * len(sorted_energy))
        threshold = sorted_energy[threshold_idx]

        peak_indices = torch.where(energy > threshold)[0]

        # create oscillation
        oscillation = torch.zeros_like(energy)

        for peak_idx in peak_indices:
            width = self.config.oscillation_width
            start = max(0, peak_idx - width)
            end = min(num_frames, peak_idx + width + 1)

            if end > start:
                osc_values = torch.sin(
                    torch.linspace(0, math.pi, end - start, device=self.device)
                )
                oscillation[start:end] += osc_values * 0.2 * energy[peak_idx]

        return oscillation

    def _compute_latent_at_frame(
        self,
        frame_idx: int,
        angle: float,
        onset_factor: float,
    ) -> torch.Tensor:
        """compute latent for specific frame given angle and onset"""
        # circular motion components
        cos_val = math.cos(angle)
        sin_val = math.sin(angle)

        noise_x = cos_val * self.base_noise_x
        noise_y = sin_val * self.base_noise_y

        # combine circular motion
        latent = noise_x + noise_y

        # add onset emphasis
        if self.onset_noise is not None and onset_factor > 0:
            latent = latent + onset_factor * self.config.onset_noise_scale * self.onset_noise

        return latent

    def _get_beat_frames(self, audio_features: AudioFeatures) -> np.ndarray:
        """get beat frames based on note type"""
        beat_frames = audio_features.beat_frames

        if self.config.note_type == "half":
            return beat_frames[::2]
        elif self.config.note_type == "whole":
            return beat_frames[::4]

        return beat_frames  # quarter notes

    def _cubic_easing(self, t: torch.Tensor) -> torch.Tensor:
        """cubic in-out easing for smooth transitions"""
        return torch.where(t < 0.5, 4 * t**3, 1 - (-2 * t + 2) ** 3 / 2)

    def reset(self) -> None:
        """reset base noise (for new song)"""
        self.base_noise_x = None
        self.base_noise_y = None
        self.onset_noise = None
