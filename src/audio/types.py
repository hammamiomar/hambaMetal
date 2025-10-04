"""
type definitions for audio analysis features
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class AudioFeatures:
    """
    immutable container for all audio features extracted from a song
    frozen for rust-like safety (no accidental mutations)
    """

    # raw audio
    y: npt.NDArray[np.float32]  # audio time series
    sr: int  # sample rate
    hop_length: int  # hop length used for analysis

    # spectral features
    mel_spec: npt.NDArray[np.float32]  # summed mel spectrogram (time,)
    mel_spec_normalized: npt.NDArray[np.float32]  # normalized mel spec

    # harmonic features
    chroma_cq: npt.NDArray[np.float32]  # chromagram (12, time)
    chroma_cq_delta: npt.NDArray[np.float32]  # chroma rate of change

    # rhythmic features
    tempo: float  # estimated tempo in bpm
    beat_frames: npt.NDArray[np.int32]  # frame indices of detected beats
    onset_frames: npt.NDArray[np.int32]  # frame indices of onsets

    # derived properties
    num_frames: int  # total number of frames
    duration_seconds: float  # audio duration in seconds
    fps: float  # frames per second (sr / hop_length)

    def __post_init__(self):
        """validate dimensions for safety"""
        # ensure all time-series features have same length
        assert len(self.mel_spec) == self.num_frames, "mel_spec length mismatch"
        assert (
            len(self.mel_spec_normalized) == self.num_frames
        ), "normalized mel_spec length mismatch"
        assert (
            self.chroma_cq.shape[1] == self.num_frames
        ), "chroma_cq time dimension mismatch"
        assert self.chroma_cq.shape[0] == 12, "chroma must have 12 pitch classes"

    def get_energy_at_frame(self, frame_idx: int) -> float:
        """get normalized energy at specific frame"""
        if 0 <= frame_idx < self.num_frames:
            return float(self.mel_spec_normalized[frame_idx])
        return 0.0

    def get_chroma_at_frame(self, frame_idx: int) -> npt.NDArray[np.float32]:
        """get 12-dimensional chroma vector at specific frame"""
        if 0 <= frame_idx < self.num_frames:
            return self.chroma_cq[:, frame_idx]
        return np.zeros(12, dtype=np.float32)

    def is_onset_frame(self, frame_idx: int) -> bool:
        """check if frame is an onset"""
        return frame_idx in self.onset_frames

    def is_beat_frame(self, frame_idx: int) -> bool:
        """check if frame is a beat"""
        return frame_idx in self.beat_frames
