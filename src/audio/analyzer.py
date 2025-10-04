"""
audio feature extraction using librosa
refactored from original NoiseVisualizer.loadSong() with type safety
"""

import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from .types import AudioFeatures
from ..config import AudioConfig


class AudioAnalyzer:
    """
    extract musical features from audio files for visualization
    parallel extraction for better performance
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()

    def analyze(self, audio_path: str | Path) -> AudioFeatures:
        """
        load and analyze audio file to extract all musical features

        args:
            audio_path: path to audio file (mp3, wav, etc.)

        returns:
            AudioFeatures: immutable container with all extracted features
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"audio file not found: {audio_path}")

        print(f"loading audio: {audio_path}")

        # suppress librosa warnings for cleaner output
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            # load audio with consistent sample rate
            y, sr = librosa.load(audio_path, sr=self.config.sr, mono=True)

        # handle very long audio (prevent oom)
        y = self._trim_long_audio(y, sr)

        print("extracting audio features...")

        # parallel feature extraction for speed
        features = (
            self._extract_features_parallel(y, sr)
            if self.config.parallel_extraction
            else self._extract_features_sequential(y, sr)
        )

        print(f"extracted {features.num_frames} frames at {features.fps:.1f} fps")
        return features

    def _trim_long_audio(
        self, y: np.ndarray, sr: int
    ) -> np.ndarray:
        """trim audio if exceeds max duration"""
        max_samples = int(self.config.max_duration_minutes * 60 * sr)

        if len(y) > max_samples:
            print(
                f"audio exceeds {self.config.max_duration_minutes} min, "
                f"trimming to first {self.config.max_duration_minutes} min"
            )
            y = y[:max_samples]

        return y

    def _extract_features_parallel(
        self, y: np.ndarray, sr: int
    ) -> AudioFeatures:
        """
        extract features using parallel execution
        faster on multi-core systems
        """
        with ThreadPoolExecutor(max_workers=4) as executor:
            # start harmonic-percussive separation
            hpss_future = executor.submit(librosa.effects.hpss, y)

            # extract mel spectrogram
            mel_future = executor.submit(
                librosa.feature.melspectrogram,
                y=y,
                sr=sr,
                n_mels=self.config.n_mels,
                hop_length=self.config.hop_length,
            )

            # extract chroma features
            chroma_future = executor.submit(
                librosa.feature.chroma_cqt,
                y=y,
                sr=sr,
                hop_length=self.config.hop_length,
                n_chroma=self.config.n_chroma,
            )

            # wait for hpss
            y_harmonic, y_percussive = hpss_future.result()

            # start onset and beat detection
            onset_env_future = executor.submit(
                librosa.onset.onset_strength,
                y=y,
                sr=sr,
                hop_length=self.config.hop_length,
            )

            beat_future = executor.submit(
                librosa.beat.beat_track,
                y=y_percussive,
                sr=sr,
                hop_length=self.config.hop_length,
            )

            # collect results
            mel_spec = np.sum(mel_future.result(), axis=0)
            chroma_cq = chroma_future.result()
            onset_env = onset_env_future.result()
            tempo, beat_frames = beat_future.result()

        # process onsets (must be sequential)
        onset_raw = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            backtrack=False,
            hop_length=self.config.hop_length,
        )
        onset_frames = librosa.onset.onset_backtrack(onset_raw, onset_env)

        return self._build_features(
            y=y,
            sr=sr,
            mel_spec=mel_spec,
            chroma_cq=chroma_cq,
            tempo=tempo,
            beat_frames=beat_frames,
            onset_frames=onset_frames,
        )

    def _extract_features_sequential(
        self, y: np.ndarray, sr: int
    ) -> AudioFeatures:
        """extract features sequentially (fallback if parallel fails)"""
        # harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # spectral features
        mel_spec_full = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.config.n_mels, hop_length=self.config.hop_length
        )
        mel_spec = np.sum(mel_spec_full, axis=0)

        # harmonic features
        chroma_cq = librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=self.config.hop_length, n_chroma=self.config.n_chroma
        )

        # rhythmic features
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=self.config.hop_length
        )
        tempo, beat_frames = librosa.beat.beat_track(
            y=y_percussive, sr=sr, hop_length=self.config.hop_length
        )

        # onset detection
        onset_raw = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            backtrack=False,
            hop_length=self.config.hop_length,
        )
        onset_frames = librosa.onset.onset_backtrack(onset_raw, onset_env)

        return self._build_features(
            y=y,
            sr=sr,
            mel_spec=mel_spec,
            chroma_cq=chroma_cq,
            tempo=tempo,
            beat_frames=beat_frames,
            onset_frames=onset_frames,
        )

    def _build_features(
        self,
        y: np.ndarray,
        sr: int,
        mel_spec: np.ndarray,
        chroma_cq: np.ndarray,
        tempo: float,
        beat_frames: np.ndarray,
        onset_frames: np.ndarray,
    ) -> AudioFeatures:
        """
        construct AudioFeatures from extracted components
        add synthetic beats if detection failed
        """
        # ensure we have enough beats (can fail on quiet audio)
        if len(beat_frames) < 2:
            print("warning: not enough beats detected, adding synthetic beats")
            num_frames = len(mel_spec)
            beat_count = max(10, num_frames // 100)
            beat_frames = np.linspace(0, num_frames - 1, beat_count, dtype=np.int32)

        # compute derived features
        mel_spec_normalized = librosa.util.normalize(mel_spec)
        chroma_cq_delta = librosa.feature.delta(chroma_cq, order=1)

        num_frames = len(mel_spec)
        duration_seconds = num_frames * self.config.hop_length / sr
        fps = sr / self.config.hop_length

        return AudioFeatures(
            y=y.astype(np.float32),
            sr=sr,
            hop_length=self.config.hop_length,
            mel_spec=mel_spec.astype(np.float32),
            mel_spec_normalized=mel_spec_normalized.astype(np.float32),
            chroma_cq=chroma_cq.astype(np.float32),
            chroma_cq_delta=chroma_cq_delta.astype(np.float32),
            tempo=float(tempo),
            beat_frames=beat_frames.astype(np.int32),
            onset_frames=onset_frames.astype(np.int32),
            num_frames=num_frames,
            duration_seconds=duration_seconds,
            fps=fps,
        )
