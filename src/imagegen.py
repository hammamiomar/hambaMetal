import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from PIL import Image
import tqdm
import moviepy.editor as mpy
import librosa

from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionXLPipeline, 
    StableDiffusionControlNetPipeline, 
    ControlNetModel, 
    TCDScheduler
)
from controlnet_aux import OpenposeDetector, MidasDetector
from huggingface_hub import hf_hub_download



class NoiseVisualizer:
    """
    A class for generating music visualizations using diffusion models.
    It analyzes audio features and generates corresponding visual elements.
    """
    
    def __init__(self, device="mps", weightType=torch.float16, seed=42069):
        """
        Initialize the visualizer with specified device and precision.
        
        Args:
            device: The device to run the model on ('cuda', 'cpu', 'mps')
            weightType: The precision to use for the model weights
            seed: Random seed for reproducibility
        """
        torch.manual_seed(seed)
        self.device = device
        self.weightType = weightType
        self.pipe = None
        self.promptPool = False
        
    def _configure_pipe_common(self, pipe):
        """Common configuration for all pipelines"""
        pipe.safety_checker = None
        pipe.set_progress_bar_config(disable=True)
        
        # Keep original precision for MPS
        return pipe
        
    def loadPipeSd(self):
        """Load standard Stable Diffusion 1.5 pipeline with HyperSD optimizations"""
        import os
        # Use local cache directory for faster loading if exists
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        
        base_model_id = "runwayml/stable-diffusion-v1-5"
        repo_name = "ByteDance/Hyper-SD"
        ckpt_name = "Hyper-SD15-1step-lora.safetensors"
        
        # Initialize pipeline with performance optimizations
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model_id, 
            torch_dtype=self.weightType,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            local_files_only=os.path.exists(os.path.join(cache_dir, "models--runwayml--stable-diffusion-v1-5"))
        )
        
        # Apply device-specific optimizations before moving to device
        if self.device.startswith("cuda"):
            # Enable memory-efficient attention on CUDA
            from diffusers.models.attention_processor import AttnProcessor2_0
            self.pipe.unet.set_attn_processor(AttnProcessor2_0())
            
            # Use channels-last memory format for better tensor core utilization
            self.pipe.unet = self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.vae = self.pipe.vae.to(memory_format=torch.channels_last)
        
        # Move to device after optimizations
        self.pipe = self.pipe.to(self.device)
        
        # Apply LoRA weights and configure
        self.pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        self.pipe.fuse_lora()
        self.pipe.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self._configure_pipe_common(self.pipe)
        
        # Set custom pipeline attributes
        self.promptPool = False
        
        # Compile model for faster inference with PyTorch 2.0+ on supported platforms
        if hasattr(torch, 'compile') and self.device != "cpu" and self.device != "mps":
            try:
                print("Compiling UNet model for faster inference...")
                self.pipe.unet = torch.compile(
                    self.pipe.unet, 
                    mode="reduce-overhead", 
                    fullgraph=False
                )
                print("UNet compilation successful")
            except Exception as e:
                print(f"Model compilation failed, falling back to standard execution: {e}")
        
    def loadPipeSdXL(self):
        """Load Stable Diffusion XL pipeline with HyperSD optimizations and performance enhancements"""
        import os
        # Use local cache directory for faster loading if exists
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        repo_name = "ByteDance/Hyper-SD"
        ckpt_name = "Hyper-SDXL-1step-lora.safetensors"
        
        # Initialize pipeline with performance optimizations
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_id, 
            torch_dtype=self.weightType,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            cache_dir=cache_dir,
            local_files_only=os.path.exists(os.path.join(cache_dir, "models--stabilityai--stable-diffusion-xl-base-1.0"))
        )
        
        # Apply device-specific optimizations before moving to device
        if self.device.startswith("cuda"):
            # Enable memory-efficient attention on CUDA
            from diffusers.models.attention_processor import AttnProcessor2_0
            self.pipe.unet.set_attn_processor(AttnProcessor2_0())
            
            # Use channels-last memory format for better tensor core utilization
            self.pipe.unet = self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.vae = self.pipe.vae.to(memory_format=torch.channels_last)
            
            # Optional: Reduce VRAM usage if needed
            # self.pipe.enable_model_cpu_offload()
        
        # Move to device after optimizations
        self.pipe = self.pipe.to(self.device)
        
        # Apply LoRA weights and configure
        self.pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        self.pipe.fuse_lora()
        self.pipe.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self._configure_pipe_common(self.pipe)
        
        # Set custom pipeline attributes
        self.promptPool = True
        
        # Compile model for faster inference with PyTorch 2.0+ on supported platforms
        if hasattr(torch, 'compile') and self.device != "cpu" and self.device != "mps":
            try:
                print("Compiling SDXL UNet model for faster inference...")
                self.pipe.unet = torch.compile(
                    self.pipe.unet, 
                    mode="reduce-overhead", 
                    fullgraph=False
                )
                print("SDXL UNet compilation successful")
            except Exception as e:
                print(f"SDXL model compilation failed, falling back to standard execution: {e}")
         
    def loadPipeSdCtrl(self, type="depth"):
        """
        Load ControlNet-enhanced Stable Diffusion pipeline
        
        Args:
            type: Type of ControlNet to use ('depth' or 'pose')
        """
        # Configure controlnet and preprocessor based on type
        if type == "depth":
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1p_sd15_depth", 
                torch_dtype=self.weightType,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.preprocessor = MidasDetector.from_pretrained("lllyasviel/Annotators")
        else:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_openpose", 
                torch_dtype=self.weightType,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.preprocessor = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            
        self.preprocessor.to(self.device)
        
        # Initialize controlled pipeline
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            controlnet=controlnet, 
            torch_dtype=self.weightType,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        # Apply LoRA weights and configure
        self.pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SD15-1step-lora.safetensors"))
        self.pipe.fuse_lora()
        self.pipe.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self._configure_pipe_common(self.pipe)
        self.promptPool = False
        
        
    
    def loadSong(self, file, hop_length):
        """
        Load and analyze an audio file to extract musical features with performance optimizations
        
        Args:
            file: Path to the audio file
            hop_length: Hop length for audio analysis (controls temporal resolution)
        """
        import concurrent.futures
        import warnings
        
        # Temporarily suppress warnings from librosa
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Load audio with consistent sample rate - use mono and specific duration if needed
        print("Loading audio file...")
        y, sr = librosa.load(file, sr=22050, mono=True)
        self.hop_length = hop_length
        self.sr = sr
        self.y = y
        
        # Handle case of very long audio file
        # Max duration of 6 minutes to avoid excessive processing
        max_frames = int((6 * 60) * sr / hop_length)
        if len(y) > max_frames * hop_length:
            print(f"Audio file is very long, limiting to first 6 minutes for performance")
            y = y[:max_frames * hop_length]
        
        print("Extracting audio features...")
        
        # Parallel audio feature extraction for better performance
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Start harmonic-percussive source separation
            hpss_future = executor.submit(librosa.effects.hpss, y)
            
            # Extract mel spectrogram concurrently
            mel_future = executor.submit(
                librosa.feature.melspectrogram,
                y=y, sr=sr, n_mels=256, hop_length=hop_length
            )
            
            # Extract chroma features concurrently
            chroma_future = executor.submit(
                librosa.feature.chroma_cqt,
                y=y, sr=sr, hop_length=hop_length, n_chroma=12
            )
            
            # Wait for HPSS to complete
            y_harmonic, y_percussive = hpss_future.result()
            
            # Start beat tracking and onset detection
            onset_env_future = executor.submit(
                librosa.onset.onset_strength,
                y=y, sr=sr, hop_length=hop_length
            )
            
            beat_future = executor.submit(
                librosa.beat.beat_track,
                y=y_percussive, sr=sr, hop_length=hop_length
            )
            
            # Get mel spectrogram results
            melSpec = mel_future.result()
            self.melSpec = np.sum(melSpec, axis=0)
            
            # Get chroma features and compute delta
            self.chroma_cq = chroma_future.result()
            self.chroma_cq_delta = librosa.feature.delta(self.chroma_cq, order=1)
            
            # Get onset envelope and beat tracking results
            oenv = onset_env_future.result()
            self.tempo, self.beat_frames = beat_future.result()
            
            # Process onsets (this needs to be sequential)
            onset_raw = librosa.onset.onset_detect(
                onset_envelope=oenv, 
                backtrack=False, 
                hop_length=self.hop_length
            )
            self.onset_bt = librosa.onset.onset_backtrack(onset_raw, oenv)
        
        # Restore warnings
        warnings.resetwarnings()
        
        # Check for empty beat frames (can happen with very quiet audio)
        if len(self.beat_frames) < 2:
            print("Warning: Not enough beats detected. Adding synthetic beats.")
            # Add synthetic beat frames if not enough were found
            frame_count = len(self.melSpec)
            beat_count = max(10, frame_count // 100)  # At least 10 beats
            self.beat_frames = np.linspace(0, frame_count-1, beat_count, dtype=np.int32)
        
        # Total number of frames for the visualization
        self.steps = self.melSpec.shape[0]
        
        # Pre-normalize the mel spectrogram for better performance
        self._normalized_melspec = librosa.util.normalize(self.melSpec)
    
    def loadVideo(self, file):
        """
        Load a video file, extract its audio for analysis, and process frames for ControlNet
        
        Args:
            file: Path to the video file
            
        Returns:
            List of processed frames for ControlNet guidance
        """
        # Load video and extract audio
        clip = mpy.VideoFileClip(file)
        clip.audio.write_audiofile("temp.wav")
        
        # Calculate appropriate hop length based on video frame rate
        self.hop_length = int(22050 / clip.fps)
        self.loadSong("temp.wav", self.hop_length)
        
        # Process video frames for ControlNet
        totalFrames = int(clip.fps * clip.duration)
        ctrlFrames = []
        for frame in tqdm.tqdm(clip.iter_frames(), total=totalFrames, desc="Processing video frames"):
            ctrlFrames.append(
                self.preprocessor(
                    Image.fromarray(frame), 
                    image_resolution=512, 
                    output_type="pil"
                )
            )
        
        # Clean up temporary audio file
        os.remove("temp.wav")
        return ctrlFrames
    
    def _get_beat_frames(self, noteType):
        """Helper to get beat frames based on note type"""
        if noteType == "half":
            return self.beat_frames[::2]
        elif noteType == "whole":
            return self.beat_frames[::4]
        return self.beat_frames  # quarter notes (default)
        
    def _cubic_easing(self, t, is_numpy=True):
        """Cubic in-out easing function for smooth transitions"""
        if is_numpy:
            return np.where(t < 0.5, 4 * t**3, 1 - (-2 * t + 2)**3 / 2)
        else:  # torch version
            return torch.where(t < 0.5, 4 * t**3, 1 - (-2 * t + 2)**3 / 2)
    
    def getEasedBeats(self, noteType):
        """
        Generate eased beat values for smooth transitions between beats
        
        Args:
            noteType: Type of note divisions to use ("quarter", "half", or "whole")
            
        Returns:
            Tensor of eased beat values
        """
        # Get appropriate beat frames based on note type
        beat_frames = self._get_beat_frames(noteType)
            
        # Initialize output array
        output_array = np.zeros(self.steps, float)

        # Set beat frame values from pre-computed normalized mel spectrogram
        output_array[beat_frames] = self._normalized_melspec[beat_frames]

        # Interpolate between beats with easing function for smooth transitions
        last_beat_idx = None
        for current_idx in beat_frames:
            if last_beat_idx is not None:
                # Calculate interpolation for all frames between beats
                for j in range(last_beat_idx + 1, current_idx):
                    t = (j - last_beat_idx) / (current_idx - last_beat_idx)
                    output_array[j] = self._cubic_easing(np.array(t))
            last_beat_idx = current_idx

        return torch.tensor(output_array, dtype=self.weightType)

    def getBeatLatentsCircle(self, distance, noteType, height=512, width=512):
        """
        Generate latent noise vectors that follow a circular path synchronized with beats
        and enhanced by musical events for more dynamic visual transitions
        
        Args:
            distance: Controls the radius of circular movement
            noteType: Type of note divisions to use ("quarter", "half", or "whole")
            height: Output image height
            width: Output image width
            
        Returns:
            Tensor of latent vectors for diffusion model input with musical responsiveness
        """
        # Set up initial noise tensors for X and Y components
        latent_channels = self.pipe.unet.in_channels
        latent_shape = (1, latent_channels, height//8, width//8)
        
        # Generate two perpendicular noise patterns for circular movement
        # Use more controlled random seed for each component for consistent motion
        torch.manual_seed(42)  # Base seed
        walkNoiseX = torch.randn(latent_shape, dtype=self.weightType, device=self.device)
        torch.manual_seed(43)  # Different seed for Y
        walkNoiseY = torch.randn(latent_shape, dtype=self.weightType, device=self.device)
        
        # Optional: Generate a third noise component for onsets
        torch.manual_seed(44)  # Different seed for onset emphasis
        onsetNoise = torch.randn(latent_shape, dtype=self.weightType, device=self.device) * 0.3

        # Use pre-computed normalized melspec for better performance
        melspec_values = torch.tensor(
            self._normalized_melspec,
            dtype=self.weightType,
            device=self.device
        )
        
        # Create onset emphasis map - stronger latent changes at musical onsets
        onset_emphasis = np.zeros(self.steps)
        onset_emphasis[self.onset_bt] = 1.0
        onset_emphasis = gaussian_filter1d(onset_emphasis, sigma=1.0)
        onset_emphasis = torch.tensor(
            onset_emphasis, 
            dtype=self.weightType,
            device=self.device
        )

        # Get beat frames based on note type
        beat_frames = self._get_beat_frames(noteType)

        # Initialize angle tensor to control circular movement
        angles = torch.zeros(self.steps, dtype=self.weightType, device=self.device)
        
        # Tempo-based adjustments
        tempo_factor = min(1.5, max(0.5, self.tempo / 120.0))  # Normalize around 120 BPM
        dynamic_distance = distance * tempo_factor  # Adjust movement range based on tempo
        
        # Enhanced easing function selection based on tempo
        if self.tempo < 80:  # Slow music
            easing_func = lambda t: self._quintic_easing(t, is_numpy=False)  # Smoother for slow music
        elif self.tempo > 160:  # Fast music
            easing_func = lambda t: self._quadratic_easing(t, is_numpy=False)  # Snappier for fast music
        else:  # Medium tempo
            easing_func = lambda t: self._cubic_easing(t, is_numpy=False)  # Default cubic
        
        # Calculate mean energy for overall scaling
        mean_energy = torch.mean(melspec_values)
        
        # Process each beat interval to create smooth angle changes
        for i in range(len(beat_frames) - 1):
            start_frame = beat_frames[i]
            end_frame = beat_frames[i + 1]
            duration = end_frame - start_frame

            if duration <= 0:
                continue  # Skip zero or negative durations

            # Scale angle change by mel spectrogram energy
            # Enhanced: Use both current and next beat energy for direction
            current_energy = melspec_values[start_frame]
            next_energy = melspec_values[end_frame]
            
            # Make movement direction respond to energy changes
            # This creates a sense of musical phrasing in the movement
            energy_ratio = next_energy / (current_energy + 1e-5)
            direction = 1.0 if energy_ratio > 1.0 else -1.0  # Move forward or backward based on energy
            
            # Enhanced: Scale movement by relative energy
            relative_energy = next_energy / (mean_energy + 1e-5)
            energy_scale = torch.clamp(relative_energy, 0.5, 2.0) 
            
            # Calculate total angle change with enhanced dynamics
            # Use detached scalar value for computation with math.pi
            total_angle_change = direction * next_energy.item() * dynamic_distance * 2 * math.pi * energy_scale.item()
            
            # Create custom easing based on segment analysis
            # Shorter segments get different curve than longer ones
            if duration < 10:  # Quick transitions
                # For short segments use an "anticipation" curve that moves slightly back before main movement
                custom_t = torch.linspace(-0.1, 1.1, steps=duration, device=self.device)
                custom_t = torch.clamp(custom_t, 0, 1)  # Clamp values to valid range
                eased_t = easing_func(custom_t)
            else:
                # Regular easing with potential hold at peak if segment is long enough
                t = torch.linspace(0, 1, steps=duration, device=self.device)
                if duration > 30:  # For very long segments, create plateau in middle
                    plateau_start = int(duration * 0.3)
                    plateau_end = int(duration * 0.7)
                    t[plateau_start:plateau_end] = 0.5  # Hold in middle section
                eased_t = easing_func(t)
            
            # Apply the angle changes to this beat interval
            # Ensure total_angle_change is a scalar or tensor, not numpy array
            angle_change = torch.tensor(total_angle_change, device=self.device, dtype=self.weightType) * eased_t
            angles[start_frame:end_frame] = angle_change

        # Handle frames after the last beat with fade-out effect
        if beat_frames[-1] < self.steps - 1:
            start_frame = beat_frames[-1]
            end_frame = self.steps - 1
            duration = end_frame - start_frame

            if duration > 0:
                # Create gentle fade-out effect
                melspec_value = melspec_values[start_frame]  # Use starting energy
                total_angle_change = melspec_value * distance * 0.5 * math.pi  # Smaller movement
                
                # Custom easing for ending - slower fade
                t = torch.linspace(0, 1, steps=duration, device=self.device)
                fade_t = torch.pow(t, 1.5)  # Power curve for fade out
                angle_change = total_angle_change * fade_t
                angles[start_frame:end_frame] = angle_change

        # Apply cumulative summation to get continuous motion instead of resets
        cumulative_angles = torch.cumsum(angles, dim=0)
        
        # Optional: Add oscillation effects at musical peaks
        # Use a percentile method directly without quantile
        # Find the value at approximately the 80th percentile
        sorted_values, _ = torch.sort(melspec_values)
        threshold_idx = int(0.8 * len(sorted_values))
        threshold = sorted_values[threshold_idx]
        peak_frames = torch.where(melspec_values > threshold)[0]
        
        if len(peak_frames) > 0:
            # Create oscillation effect at energy peaks
            oscillation = torch.zeros_like(cumulative_angles)
            for peak in peak_frames:
                width = 5  # Oscillation spread
                start_idx = max(0, peak - width)
                end_idx = min(len(oscillation), peak + width + 1)
                if end_idx > start_idx:
                    oscillation[start_idx:end_idx] = torch.sin(
                        torch.linspace(0, math.pi, end_idx - start_idx, device=self.device)
                    ) * 0.2 * melspec_values[peak]
            
            # Add oscillation to the main movement
            cumulative_angles = cumulative_angles + oscillation

        # Convert angles to X/Y coordinates using sine/cosine
        walkScaleX = torch.cos(cumulative_angles).view(-1, 1, 1, 1)  # Reshape for broadcasting
        walkScaleY = torch.sin(cumulative_angles).view(-1, 1, 1, 1)
        
        # Create onset emphasis factor (view for broadcasting)
        onset_factor = onset_emphasis.view(-1, 1, 1, 1)
        
        # Apply the circular movement to the noise vectors
        # With enhanced onset emphasis using the extra noise component
        noiseX = walkScaleX * walkNoiseX
        noiseY = walkScaleY * walkNoiseY
        noiseOnset = onset_factor * onsetNoise  # Emphasis noise at onsets
        
        # Combine components to create the final latent vectors
        # This creates core circular path plus additional variation at musical events
        return noiseX + noiseY + noiseOnset
    
    def _quadratic_easing(self, t, is_numpy=True):
        """Quadratic in-out easing function - snappier transitions"""
        if is_numpy:
            return np.where(t < 0.5, 2 * t**2, 1 - (-2 * t + 2)**2 / 2)
        else:  # torch version
            return torch.where(t < 0.5, 2 * t**2, 1 - (-2 * t + 2)**2 / 2)
    
    def _quintic_easing(self, t, is_numpy=True):
        """Quintic in-out easing function - smoother transitions"""
        if is_numpy:
            return np.where(t < 0.5, 16 * t**5, 1 - (-2 * t + 2)**5 / 32)
        else:  # torch version
            return torch.where(t < 0.5, 16 * t**5, 1 - (-2 * t + 2)**5 / 32)

    def getFPS(self):
        """Calculate frames per second based on sampling rate and hop length"""
        # Check if sr or hop_length is None or zero to avoid errors
        if not hasattr(self, 'sr') or not hasattr(self, 'hop_length') or self.sr is None or self.hop_length is None or self.hop_length == 0:
            print("Warning: Cannot calculate FPS, missing or invalid sr or hop_length values")
            return 60  # Return default fps
        return round(self.sr / self.hop_length)
    
    def slerp(self, embed1, embed2, alpha):
        """
        Spherical linear interpolation between two embeddings
        
        Args:
            embed1: First embedding tensor
            embed2: Second embedding tensor
            alpha: Interpolation factor (0.0 to 1.0)
            
        Returns:
            Interpolated embedding
        """
        # Normalize embeddings for proper spherical interpolation
        embed1_norm = embed1 / torch.norm(embed1, dim=-1, keepdim=True)
        embed2_norm = embed2 / torch.norm(embed2, dim=-1, keepdim=True)

        # Calculate angle between embeddings
        dot_product = torch.sum(embed1_norm * embed2_norm, dim=-1, keepdim=True)
        omega = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
        sin_omega = torch.sin(omega)
        
        # Handle special case of parallel vectors
        if torch.any(sin_omega == 0):
            return (1.0 - alpha) * embed1 + alpha * embed2  # Fall back to linear interpolation

        # Perform spherical interpolation
        interp_embed = (
            torch.sin((1.0 - alpha) * omega) / sin_omega * embed1 +
            torch.sin(alpha * omega) / sin_omega * embed2
        )
        return interp_embed 
    
    def getPromptEmbedsClip(self, basePrompt, targetPromptChromaScale, alpha=0.5, smoothing_factor=0.3, power_scale=2.0):
        """
        Generate prompt embeddings that respond to chroma features in the music
        
        Args:
            basePrompt: Base prompt to use for all frames
            targetPromptChromaScale: List of target prompts for each chroma
            alpha: Strength of chroma influence
            smoothing_factor: Temporal smoothing factor to avoid jitter
            power_scale: Exponent to emphasize stronger chroma components
            
        Returns:
            Tensor of prompt embeddings for each frame
        """
        # Convert chroma data to tensor
        chroma = torch.tensor(self.chroma_cq.T, device=self.device)  # (numFrames, 12)
        
        # Emphasize stronger chroma components and normalize
        scaled_powers = torch.pow(torch.clamp(chroma, min=0), power_scale)
        weights = F.softmax(scaled_powers, dim=1)  # Normalize across chroma dimension
        
        # Initialize or retrieve previous weights for temporal smoothing
        if not hasattr(self, 'prev_weights'):
            self.prev_weights = weights
        
        # Apply exponential smoothing across frames
        smoothed = (1 - smoothing_factor) * self.prev_weights + smoothing_factor * weights
        self.prev_weights = smoothed
        
        # For SDXL pipeline that uses pooled embeddings
        if self.promptPool:
            # Encode base and target prompts using SDXL encoder
            baseEmbeds, _, baseEmbedsPooled, _ = self.pipe.encode_prompt(
                prompt=basePrompt,
                prompt_2=basePrompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )
            targetEmbeds, _, targetEmbedsPooled, _ = self.pipe.encode_prompt(
                prompt=targetPromptChromaScale,
                prompt_2=targetPromptChromaScale,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )
            
            # Remove batch dimension
            baseEmbeds = baseEmbeds.squeeze(0)
            baseEmbedsPooled = baseEmbedsPooled.squeeze(0)
            
            # Process and interpolate embeddings for each frame
            interpolatedEmbedsAll = []
            interpolatedEmbedsAllPooled = []
            
            for frame in range(len(smoothed)):
                # Scale weights by alpha to control influence
                frame_weights = smoothed[frame] * alpha
                
                # Start with base embedding weighted by remaining influence
                base_weight = 1.0 - frame_weights.sum()
                interpolatedEmbed = base_weight * baseEmbeds
                interpolatedEmbedPooled = base_weight * baseEmbedsPooled
                
                # Add weighted contributions from each chroma dimension
                for i in range(12):  # 12 chromatic notes
                    if frame_weights[i] > 0:  # Skip processing insignificant weights
                        target_embed = targetEmbeds[i]
                        target_embed_pooled = targetEmbedsPooled[i]
                        
                        # Use spherical interpolation for better blending
                        interpolatedEmbed = self.slerp(interpolatedEmbed, target_embed, frame_weights[i])
                        interpolatedEmbedPooled = self.slerp(
                            interpolatedEmbedPooled, target_embed_pooled, frame_weights[i]
                        )
                
                interpolatedEmbedsAll.append(interpolatedEmbed)
                interpolatedEmbedsAllPooled.append(interpolatedEmbedPooled)
            
            # Stack frame embeddings into a single tensor
            return torch.stack(interpolatedEmbedsAll), torch.stack(interpolatedEmbedsAllPooled)
        
        # For standard SD pipeline that doesn't use pooled embeddings
        else:
            # Encode base and target prompts
            baseEmbeds, _ = self.pipe.encode_prompt(
                prompt=basePrompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )
            targetEmbeds, _ = self.pipe.encode_prompt(
                prompt=targetPromptChromaScale,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False
            )
            
            # Remove batch dimension and create frame containers
            baseEmbeds = baseEmbeds.squeeze(0)
            interpolatedEmbedsAll = []
            
            # Process each frame
            for frame in range(len(smoothed)):
                frame_weights = smoothed[frame] * alpha
                
                # Start with weighted base embedding
                base_weight = 1.0 - frame_weights.sum()
                interpolatedEmbed = base_weight * baseEmbeds
                
                # Add contributions from each chroma dimension
                for i in range(12):
                    if frame_weights[i] > 0:
                        target_embed = targetEmbeds[i]
                        interpolatedEmbed = self.slerp(interpolatedEmbed, target_embed, frame_weights[i])
                
                interpolatedEmbedsAll.append(interpolatedEmbed)
            
            # Stack frame embeddings into a single tensor
            return torch.stack(interpolatedEmbedsAll)

    def getPromptEmbedsSlerpSum(self, basePrompt, targetPromptChromaScale, alpha):
        chroma = torch.tensor(self.chroma_cq.T, device=self.device)  # (numFrames, 12)
        numFrames, numChroma = chroma.shape  # numFrames and 12 chromatic notes

        # Step 1: Get Alphas per frame
        chromaPower = torch.sum(chroma, dim=1)
        chromaPowerNorm = (chromaPower - torch.min(chromaPower)) / (torch.max(chromaPower) - torch.min(chromaPower))
        chromaPowerNormMean = torch.mean(chromaPowerNorm)
        alphas = alpha + (chromaPowerNorm - chromaPowerNormMean)
        alphas = torch.clamp(alphas, 0, 1)  # Clamp alphas to the range [0, 1]

        # Step 2: Create base and target embeddings
        if self.promptPool:
            baseEmbeds, _, baseEmbedsPooled, _ = self.pipe.encode_prompt(
                prompt=basePrompt, prompt_2=basePrompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )
            targetEmbeds, _, targetEmbedsPooled, _ = self.pipe.encode_prompt(
                prompt=targetPromptChromaScale, prompt_2=targetPromptChromaScale, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )
        else:
            baseEmbeds, _ = self.pipe.encode_prompt(
                prompt=basePrompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )
            targetEmbeds, _ = self.pipe.encode_prompt(
                prompt=targetPromptChromaScale, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )

        # baseEmbeds shape: (1, 77, 768), targetEmbeds shape: (12, 77, 768)

        # Step 3: Compute the weighted sum of target embeddings for each frame using einsum for simplicity and efficiency
        # chroma shape [numFrames, 12], targetEmbeds shape [12, 77, 768]
        weighted_sums = torch.einsum('fi,ijk->fjk', chroma, targetEmbeds)  # Resulting shape [numFrames, 77, 768]

        # Step 4: Define SLERP function in torch
        def slerp(t, v0, v1):
            """ Spherical linear interpolation between two vectors v0 and v1 with interpolation factor t. """
            v0_norm = F.normalize(v0, dim=-1)
            v1_norm = F.normalize(v1, dim=-1)
            dot_product = torch.sum(v0_norm * v1_norm, dim=-1, keepdim=True)
            omega = torch.acos(torch.clamp(dot_product, -1 + 1e-7, 1 - 1e-7))
            sin_omega = torch.sin(omega)
            slerped = (torch.sin((1.0 - t) * omega) / sin_omega * v0 +
                    torch.sin(t * omega) / sin_omega * v1)
            return slerped

        # Step 5: SLERP from base embedding to the weighted sums for each frame
        alphas = alphas.unsqueeze(1).unsqueeze(2)  # Reshape alphas to [numFrames, 1, 1] for broadcasting
        base_embeds_expanded = baseEmbeds.expand(numFrames, -1, -1)  # Expand baseEmbeds to match frame count
        interpolated_embeds = slerp(alphas, base_embeds_expanded, weighted_sums)  # SLERP to weighted sums

        # Handle pooled embeddings if self.promptPool is True
        if self.promptPool:
            weighted_sums_pooled = torch.einsum('fi,ik->fk', chroma, targetEmbedsPooled)  # [numFrames, 768]
            base_embeds_pooled_expanded = baseEmbedsPooled.expand(numFrames, -1)  # Expand base pooled embeddings
            interpolated_embeds_pooled = slerp(alphas.squeeze(2), base_embeds_pooled_expanded, weighted_sums_pooled)

            return interpolated_embeds, interpolated_embeds_pooled  # Return both regular and pooled embeddings

        return interpolated_embeds  # If promptPool is False, return only the regular embeddings
        
    def getPromptEmbedsOnsetFocus(self, basePrompt, targetPromptChromaScale, alpha=0.5, sigma=2):
        chroma = self.chroma_cq.T  # shape: (numFrames, 12)
        numFrames = chroma.shape[0]

        top_chromaOnset = np.argmax(np.abs(self.chroma_cq_delta), axis=0)
        # Onset frames and dominant chromas
        onset_frames = self.onset_bt
        dominant_chromas = top_chromaOnset[onset_frames]

        # Initialize alphas and dominant chroma per frame
        alphas = np.zeros(numFrames)
        dominant_chroma_per_frame = np.full(numFrames, -1, dtype=int)

        # For each onset interval
        for i in range(len(onset_frames) - 1):
            start_frame = onset_frames[i]
            end_frame = onset_frames[i + 1]
            dominant_chroma = dominant_chromas[i]
            
            if start_frame == end_frame:
                end_frame+=1

            # Extract chroma magnitudes of the dominant chroma
            chroma_magnitudes = chroma[start_frame:end_frame, dominant_chroma]

            # Normalize chroma magnitudes to [0, alpha_max] so that the chroma can experience full visual effect
            min_val = chroma_magnitudes.min()
            max_val = chroma_magnitudes.max()
            if max_val - min_val > 0:
                normalized_magnitudes = (chroma_magnitudes - min_val) / (max_val - min_val)
            else:
                normalized_magnitudes = np.zeros_like(chroma_magnitudes)

            alphas_interval = normalized_magnitudes * alpha
            alphas[start_frame:end_frame] = alphas_interval

            # Assign dominant chroma to frames
            dominant_chroma_per_frame[start_frame:end_frame] = dominant_chroma

        # Handle frames after the last onset
        start_frame = onset_frames[-1]
        dominant_chroma = dominant_chromas[-1]
        end_frame = numFrames

        chroma_magnitudes = chroma[start_frame:end_frame, dominant_chroma]
        min_val = chroma_magnitudes.min()
        max_val = chroma_magnitudes.max()
        if max_val - min_val > 0:
            normalized_magnitudes = (chroma_magnitudes - min_val) / (max_val - min_val)
        else:
            normalized_magnitudes = np.zeros_like(chroma_magnitudes)

        alphas_interval = normalized_magnitudes * alpha
        alphas[start_frame:end_frame] = alphas_interval
        dominant_chroma_per_frame[start_frame:end_frame] = dominant_chroma

        # Apply temporal smoothing to alphas (optional)
        alphas = gaussian_filter1d(alphas, sigma=sigma)

        if self.promptPool:
            baseEmbeds,baseNegativeEmbeds,baseEmbedsPooled,baseNegativeEmbedsPooled = self.pipe.encode_prompt(prompt=basePrompt,prompt_2=basePrompt,
                                                                                                              device=self.device,num_images_per_prompt=1,do_classifier_free_guidance=False)
            
            targetEmbeds,targetNegativeEmbeds,targetEmbedsPooled,targetNegativeEmbedsPooled = self.pipe.encode_prompt(prompt=targetPromptChromaScale,prompt_2=targetPromptChromaScale,
                                                                                                                      device=self.device,num_images_per_prompt=1,do_classifier_free_guidance=False)
            
            baseEmbeds = baseEmbeds.squeeze(0)
            baseEmbedsPooled = baseEmbedsPooled.squeeze(0)
            # Initialize interpolatedEmbedsAll
            interpolatedEmbedsAll = []
            interpolatedEmbedsAllPooled = []

            # For each frame
            for frame in range(numFrames):
                
                alphaFrame = alphas[frame]
                dominant_chroma = dominant_chroma_per_frame[frame]
                
                target_embed = targetEmbeds[dominant_chroma]  # shape: (seq_len, hidden_size)
                target_embedPooled = targetEmbedsPooled[dominant_chroma]
                
                interpolatedEmbed = self.slerp(baseEmbeds, target_embed, alphaFrame)
                interpolatedEmbedPooled = self.slerp(baseEmbedsPooled, target_embedPooled, alphaFrame)
                
                interpolatedEmbedsAll.append(interpolatedEmbed)  # shape: (1, seq_len, hidden_size)
                interpolatedEmbedsAllPooled.append(interpolatedEmbedPooled)

            interpolatedEmbeds = torch.stack(interpolatedEmbedsAll)  # shape: (numFrames, 1, seq_len, hidden_size)
            interpolatedEmbedsPooled = torch.stack(interpolatedEmbedsAllPooled)
            
            return interpolatedEmbeds, interpolatedEmbedsPooled
        else:
            baseEmbeds,baseNegativeEmbeds = self.pipe.encode_prompt(prompt=basePrompt,device=self.device,num_images_per_prompt=1,do_classifier_free_guidance=False)
            
            targetEmbeds,targetNegativeEmbeds = self.pipe.encode_prompt(prompt=targetPromptChromaScale, device=self.device,num_images_per_prompt=1,do_classifier_free_guidance=False)

            baseEmbeds = baseEmbeds.squeeze(0)
            # Initialize interpolatedEmbedsAll
            interpolatedEmbedsAll = []

            # For each frame
            for frame in range(numFrames):
                
                alphaFrame = alphas[frame]
                dominant_chroma = dominant_chroma_per_frame[frame]
                target_embed = targetEmbeds[dominant_chroma]  # shape: (seq_len, hidden_size)
                interpolatedEmbed = self.slerp(baseEmbeds, target_embed, alphaFrame)
                interpolatedEmbedsAll.append(interpolatedEmbed)  # shape: (1, seq_len, hidden_size)

            interpolatedEmbeds = torch.stack(interpolatedEmbedsAll)  # shape: (numFrames, 1, seq_len, hidden_size)

            return interpolatedEmbeds
        
        
    def getPromptEmbedsCum(self, basePrompt, targetPromptChromaScale, alpha=0.5, sigma_time=2, sigma_chroma=1, number_of_chromas_focus=6,
                           num_prompt_shuffles=4):
        """
        Advanced prompt embedding generator that responds to musical events with enhanced transitions
        
        This method creates prompt embeddings that change in response to musical events (onsets),
        harmonic changes (chromas), and energy variations, with enhanced transitions between states
        
        Args:
            basePrompt: Base prompt used for all frames
            targetPromptChromaScale: List of target prompts for each chroma (musical note)
            alpha: Overall strength of the musical influence (0.0-1.0)
            sigma_time: Temporal smoothing factor (higher = smoother transitions)
            sigma_chroma: Smoothing between different chroma influences
            number_of_chromas_focus: Number of chroma dimensions to consider
            num_prompt_shuffles: Times to shuffle prompt embeddings for variety
            
        Returns:
            Tensor of prompt embeddings synchronized with musical features
        """
        # Extract chromagram and convert to frame-oriented format
        chroma = self.chroma_cq.T  # shape: (numFrames, 12)
        numFrames = chroma.shape[0]
        number_of_chromas = number_of_chromas_focus
        
        # Get energy profile for influence scaling
        energy_profile = librosa.util.normalize(self.melSpec)
        
        # Get rate of change for each chroma (for detecting harmonic changes)
        chroma_cq_delta_abs = np.abs(self.chroma_cq_delta)
        
        # Enhance onset detection for better musical event responsiveness
        onset_strength = np.zeros(numFrames)
        onset_strength[self.onset_bt] = 1.0
        onset_strength = gaussian_filter1d(onset_strength, sigma=1.0)
        
        # Find the most active chromas at each frame based on rate of change
        top_chromas = np.argsort(-chroma_cq_delta_abs, axis=0)[:number_of_chromas, :]
        
        # Get the most significant chromas at onset points
        onset_frames = self.onset_bt
        top_chromas_at_onsets = top_chromas[:, self.onset_bt]
        
        # Initialize influence values and chroma indices for each frame
        alphas = np.zeros((numFrames, number_of_chromas))
        chromas_per_frame = np.full((numFrames, number_of_chromas), -1, dtype=int)
        
        # Enhanced method: Process each onset interval with musical awareness
        for i in range(len(onset_frames) - 1):
            start_frame = onset_frames[i]
            end_frame = onset_frames[i + 1]
            chroma_indices = top_chromas_at_onsets[:, i]
            
            # Ensure valid interval
            if start_frame == end_frame:
                end_frame += 1
                
            # Get frame interval length for this musical segment
            interval_length = end_frame - start_frame
            
            # Extract chroma magnitudes for this musical segment
            chroma_magnitudes = chroma[start_frame:end_frame, chroma_indices]
            
            # Calculate segment energy for dynamic influence
            segment_energy = energy_profile[start_frame:end_frame]
            segment_energy = segment_energy.reshape(-1, 1)  # For broadcasting
            
            # Enhanced: Apply amplitude envelope to create musical articulation
            # Create natural attack-decay-sustain-release envelope based on segment length
            if interval_length > 3:
                # Define envelope points (percentages of interval)
                attack_point = max(1, int(interval_length * 0.1))  # 10% attack
                decay_point = max(2, int(interval_length * 0.2))   # 20% decay
                release_point = max(3, int(interval_length * 0.8)) # 80% release start
                
                # Create the ADSR envelope
                envelope = np.ones(interval_length)
                # Attack phase: 0 to peak
                envelope[:attack_point] = np.linspace(0.2, 1.0, attack_point)
                # Decay phase: peak to sustain level
                if decay_point > attack_point:
                    envelope[attack_point:decay_point] = np.linspace(1.0, 0.7, decay_point - attack_point)
                # Release phase: sustain to end
                if interval_length > release_point:
                    envelope[release_point:] = np.linspace(0.7, 0.3, interval_length - release_point)
                
                # Apply envelope to modulate influence
                envelope = envelope.reshape(-1, 1)  # For broadcasting
                chroma_magnitudes = chroma_magnitudes * envelope * segment_energy
            else:
                # For very short segments, use simpler scaling with energy
                chroma_magnitudes = chroma_magnitudes * segment_energy
            
            # Normalize influences within segment
            magnitudes_sum = chroma_magnitudes.sum(axis=1, keepdims=True) + 1e-8
            alpha_values = (chroma_magnitudes / magnitudes_sum)
            
            # Assign influence values and chroma indices
            alphas[start_frame:end_frame, :] = alpha_values
            chromas_per_frame[start_frame:end_frame, :] = chroma_indices.reshape(1, number_of_chromas)
        
        # Handle frames after the last onset
        if len(onset_frames) > 0:  # Only process if we have onsets
            start_frame = onset_frames[-1]
            end_frame = numFrames
            chroma_indices = top_chromas_at_onsets[:, -1]
            
            if end_frame > start_frame:
                # Create fade-out for ending
                interval_length = end_frame - start_frame
                chroma_magnitudes = chroma[start_frame:end_frame, chroma_indices]
                
                # Smooth fade out for ending
                fade_out = np.linspace(1.0, 0.2, interval_length).reshape(-1, 1)
                segment_energy = energy_profile[start_frame:end_frame].reshape(-1, 1)
                chroma_magnitudes = chroma_magnitudes * fade_out * segment_energy
                
                # Normalize
                magnitudes_sum = chroma_magnitudes.sum(axis=1, keepdims=True) + 1e-8
                alpha_values = (chroma_magnitudes / magnitudes_sum)
                
                alphas[start_frame:end_frame, :] = alpha_values
                chromas_per_frame[start_frame:end_frame, :] = chroma_indices.reshape(1, number_of_chromas)
        
        # Apply multi-dimensional smoothing for natural transitions
        # This creates a perceptually pleasing flow between musical events
        alphas_smoothed = gaussian_filter(
            alphas, 
            sigma=(sigma_time, sigma_chroma),
            mode='reflect',
            truncate=3.0  # Use 3 standard deviations for higher quality
        )
        
        # Scale by overall alpha value while preserving relative weights
        magnitudes_sum = alphas_smoothed.sum(axis=1, keepdims=True) + 1e-8
        alphas_normalized = (alphas_smoothed / magnitudes_sum) * alpha
        
        # Apply onset enhancement: increase influence at strong musical events
        onset_boost = np.clip(onset_strength.reshape(-1, 1) * 0.5, 0, 0.3)
        alphas_normalized = np.clip(alphas_normalized + onset_boost, 0, alpha)
        
        # Final post-processing to ensure constraints
        # Re-normalize to ensure we don't exceed alpha
        row_sums = alphas_normalized.sum(axis=1, keepdims=True)
        over_alpha = row_sums > alpha
        if np.any(over_alpha):
            # Only re-normalize rows that exceed alpha
            alphas_normalized[over_alpha.flatten()] = (
                alphas_normalized[over_alpha.flatten()] / 
                row_sums[over_alpha][0] * alpha
            )
        
        # Update the alphas array with the enhanced values
        alphas = alphas_normalized

        if self.promptPool:
            baseEmbeds,baseNegativeEmbeds,baseEmbedsPooled,baseNegativeEmbedsPooled = self.pipe.encode_prompt(prompt=basePrompt,prompt_2=basePrompt,
                                                                                                              device=self.device,num_images_per_prompt=1,do_classifier_free_guidance=False)
            
            targetEmbeds,targetNegativeEmbeds,targetEmbedsPooled,targetNegativeEmbedsPooled = self.pipe.encode_prompt(prompt=targetPromptChromaScale,prompt_2=targetPromptChromaScale,
                                                                                                                      device=self.device,num_images_per_prompt=1,do_classifier_free_guidance=False)
            
            baseEmbeds = baseEmbeds.squeeze(0)
            baseEmbedsPooled = baseEmbedsPooled.squeeze(0)
            # Initialize interpolatedEmbedsAll
            interpolatedEmbedsAll = []
            interpolatedEmbedsAllPooled = []
            
            shuffle_points = [int(i * numFrames / num_prompt_shuffles) for i in range(1, num_prompt_shuffles)]  # Calculate exact shuffle points
                    # For each frame

            for frame in range(numFrames):

                if frame in shuffle_points:
                    targetEmbeds = targetEmbeds[torch.randperm(targetEmbeds.size(0))]# Shuffle along dimension 0
                    targetEmbedsPooled = targetEmbedsPooled[torch.randperm(targetEmbedsPooled.size(0))]# Shuffle along dimension 0
                            
                            
                alpha_values = alphas[frame, :]  # shape: (number_of_chromas,)
                total_alpha = alpha_values.sum() # TODO IS THIS NEEEDED??? 
                if total_alpha > alpha:
                    alpha_values = (alpha_values / total_alpha) * alpha
                    total_alpha = alpha

                base_alpha = 1.0 - total_alpha
                chroma_indices = chromas_per_frame[frame, :]  # shape: (number_of_chromas,)  what chromas to show in this frame

                # Start with baseEmbeds multiplied by base_alpha
                interpolatedEmbed = base_alpha * baseEmbeds
                interpolatedEmbedPooled = base_alpha * baseEmbedsPooled

                # Add contributions from each target chroma
                
                for n in range(number_of_chromas):
                    target_embed = targetEmbeds[chroma_indices[n]]  # shape: (seq_len, hidden_size)
                    interpolatedEmbed += alpha_values[n] * target_embed 
                    
                    target_embedPooled = targetEmbedsPooled[chroma_indices[n]]  # shape: (seq_len, hidden_size)
                    interpolatedEmbedPooled += alpha_values[n] * target_embedPooled
                
                interpolatedEmbedsAll.append(interpolatedEmbed)  # shape: (1, seq_len, hidden_size)
                interpolatedEmbedsAllPooled.append(interpolatedEmbedPooled)  # shape: (1, seq_len, hidden_size)

            interpolatedEmbeds = torch.stack(interpolatedEmbedsAll)  # shape: (numFrames, seq_len, hidden_size)
            interpolatedEmbedsPooled = torch.stack(interpolatedEmbedsAllPooled)

            return interpolatedEmbeds, interpolatedEmbedsPooled

        
        else:
            baseEmbeds,baseNegativeEmbeds = self.pipe.encode_prompt(prompt=basePrompt,device=self.device,num_images_per_prompt=1,do_classifier_free_guidance=False)
            
            targetEmbeds,targetNegativeEmbeds = self.pipe.encode_prompt(prompt=targetPromptChromaScale, device=self.device,num_images_per_prompt=1,do_classifier_free_guidance=False)

            baseEmbeds = baseEmbeds.squeeze(0)
            # Initialize interpolatedEmbedsAll
            interpolatedEmbedsAll = []

            
            shuffle_points = [int(i * numFrames / num_prompt_shuffles) for i in range(1, num_prompt_shuffles)]  # Calculate exact shuffle points
                    # For each frame

            for frame in range(numFrames):

                if frame in shuffle_points:
                    targetEmbeds = targetEmbeds[torch.randperm(targetEmbeds.size(0))]# Shuffle along dimension 0
                    print("shuffled at frame ",frame)
                            
                alpha_values = alphas[frame, :]  # shape: (number_of_chromas,)
                total_alpha = alpha_values.sum()
                if total_alpha > alpha:
                    alpha_values = (alpha_values / total_alpha) * alpha
                    total_alpha = alpha

                base_alpha = 1.0 - total_alpha
                chroma_indices = chromas_per_frame[frame, :]  # shape: (number_of_chromas,)  what chromas to show in this frame

                # Start with baseEmbeds multiplied by base_alpha
                interpolatedEmbed = base_alpha * baseEmbeds

                # Add contributions from each target chroma
                
                for n in range(number_of_chromas):
                    target_embed = targetEmbeds[chroma_indices[n]]  # shape: (seq_len, hidden_size)
                    interpolatedEmbed += alpha_values[n] * target_embed 
                
                interpolatedEmbedsAll.append(interpolatedEmbed)  # shape: (1, seq_len, hidden_size)

            interpolatedEmbeds = torch.stack(interpolatedEmbedsAll)  # shape: (numFrames, seq_len, hidden_size)

            return interpolatedEmbeds

    def _generate_frames_batch(self, **kwargs):
        """Common method for batched frame generation with memory optimization"""
        import gc
        allFrames = []
        num_frames = kwargs.pop('num_frames')
        batch_size = kwargs.pop('batch_size')
        
        # Set larger batch size for CPU (no VRAM limitations)
        if self.device == "cpu" and batch_size < 16:
            batch_size = min(16, num_frames)
            print(f"Using larger batch size {batch_size} for CPU processing")
        
        # Prefetch tensors outside the loop to avoid repeated dict operations
        tensor_keys = []
        tensor_values = []
        static_inputs = {}
        
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) or key == 'image':
                tensor_keys.append(key)
                tensor_values.append(value)
            else:
                static_inputs[key] = value
        
        # Use no_grad for efficient inference
        with torch.no_grad():
            # Process in batches with progress bar
            for i in tqdm.tqdm(range(0, num_frames, batch_size), desc="Generating frames"):
                # Get batch indices
                batch_end = min(i + batch_size, num_frames)
                batch_slice = slice(i, batch_end)
                
                # Prepare batch inputs efficiently
                batch_inputs = static_inputs.copy()  # Static inputs don't change between batches
                
                # Add dynamic tensor inputs for this batch
                for key, value in zip(tensor_keys, tensor_values):
                    batch_inputs[key] = value[batch_slice]
                
                # Generate frames for this batch
                frames = self.pipe(**batch_inputs, output_type="pil").images
                allFrames.extend(frames)
                
                # Manual GPU memory cleanup
                if self.device != "cpu" and i % (batch_size * 4) == 0:
                    if self.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    gc.collect()
        
        # Final cleanup
        if self.device != "cpu":
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()
            
        return allFrames
    
    def getVisuals(self, latents, promptEmbeds, num_inference_steps=1, batch_size=2, guidance_scale=0):
        """
        Generate visual frames using standard Stable Diffusion pipeline
        
        Args:
            latents: Pre-generated latent vectors
            promptEmbeds: Pre-encoded prompt embeddings
            num_inference_steps: Number of denoising steps (lower = faster, less quality)
            batch_size: Number of frames to process in parallel
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            List of PIL Image frames
        """
        # Ensure inputs are on the correct device
        latents = latents.to(self.device)
        promptEmbeds = promptEmbeds.to(self.device)
        
        return self._generate_frames_batch(
            prompt_embeds=promptEmbeds,
            latents=latents,
            guidance_scale=guidance_scale,
            eta=1.0,
            num_inference_steps=num_inference_steps,
            num_frames=self.steps,
            batch_size=batch_size
        )
    
    def getVisualsPooled(self, latents, promptEmbeds, promptEmbedsPooled, num_inference_steps=4, batch_size=1, guidance_scale=3):
        """
        Generate visual frames using SDXL pipeline with pooled embeddings
        
        Args:
            latents: Pre-generated latent vectors
            promptEmbeds: Pre-encoded prompt embeddings
            promptEmbedsPooled: Pooled prompt embeddings for SDXL
            num_inference_steps: Number of denoising steps (lower = faster, less quality)
            batch_size: Number of frames to process in parallel
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            List of PIL Image frames
        """
        # Ensure inputs are on the correct device
        latents = latents.to(self.device)
        promptEmbeds = promptEmbeds.to(self.device)
        promptEmbedsPooled = promptEmbedsPooled.to(self.device)
        
        return self._generate_frames_batch(
            prompt_embeds=promptEmbeds,
            pooled_prompt_embeds=promptEmbedsPooled,
            latents=latents,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_frames=self.steps,
            batch_size=batch_size
        )
        
    def getVisualsCtrl(self, latents, promptEmbeds, ctrlFrames, num_inference_steps=1, batch_size=2, guidance_scale=0, width=512, height=512):
        """
        Generate visual frames using ControlNet-guided Stable Diffusion
        
        Args:
            latents: Pre-generated latent vectors
            promptEmbeds: Pre-encoded prompt embeddings
            ctrlFrames: List of control images for guiding generation
            num_inference_steps: Number of denoising steps
            batch_size: Number of frames to process in parallel
            guidance_scale: Classifier-free guidance scale
            width: Output image width
            height: Output image height
            
        Returns:
            List of PIL Image frames
        """
        # Limit processing to available control frames
        num_frames = len(ctrlFrames)
        
        # Ensure inputs are on the correct device
        latents = latents[:num_frames].to(self.device)
        promptEmbeds = promptEmbeds[:num_frames].to(self.device)
        
        return self._generate_frames_batch(
            prompt_embeds=promptEmbeds,
            latents=latents,
            image=ctrlFrames,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            eta=1.0,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            batch_size=batch_size
        )
    
def create_mp4_from_pil_images(image_array, output_path, song, fps):
    """
    Creates an MP4 video at the specified frame rate from an array of PIL images.

    :param image_array: List of PIL images to be used as frames in the video.
    :param output_path: Path where the output MP4 file will be saved.
    :param fps: Frames per second for the output video. Default is 60.
    """
    # Set default fps to 60 if None
    if fps is None:
        print("Warning: fps was None, defaulting to 60 fps")
        fps = 60
        
    # Convert PIL images to moviepy's ImageClip format
    clips = [mpy.ImageClip(np.array(img)).set_duration(1/fps) for img in image_array]

    # Concatenate all the clips into a single video clip
    video = mpy.concatenate_videoclips(clips, method="compose")

    video = video.set_audio(mpy.AudioFileClip(song, fps=44100))
    # Write the result to a file
    video.write_videofile(output_path, fps=fps, audio_codec='aac')

def create_quad_mp4_from_pil_images(top_left, top_right, bottom_left, bottom_right, output_path, song, fps):
    """
    Creates a 2x2 grid video from four different image sequences
    
    Args:
        top_left: List of PIL images for top left quadrant
        top_right: List of PIL images for top right quadrant
        bottom_left: List of PIL images for bottom left quadrant
        bottom_right: List of PIL images for bottom right quadrant 
        output_path: Path where the output MP4 file will be saved
        song: Path to the audio file to use as soundtrack
        fps: Frames per second for the output video. Default is 60 if None.
    """
    # Set default fps to 60 if None
    if fps is None:
        print("Warning: fps was None in quad video, defaulting to 60 fps")
        fps = 60
    # Find shortest sequence length to ensure all quadrants have the same number of frames
    min_frames = min(len(top_left), len(top_right), len(bottom_left), len(bottom_right))
    
    # Trim all sequences to the same length
    quadrants = [
        top_left[:min_frames],
        top_right[:min_frames],
        bottom_left[:min_frames],
        bottom_right[:min_frames]
    ]
    
    # Get dimensions from first image (assuming all have the same size)
    width, height = top_left[0].size
    
    # Create combined frames by arranging in a 2x2 grid
    combined_frames = []
    for i in range(min_frames):
        # Create a canvas for the combined image
        combined_img = Image.new('RGB', (width * 2, height * 2))
        
        # Add each quadrant to its position
        positions = [(0, 0), (width, 0), (0, height), (width, height)]
        for quadrant, pos in zip(quadrants, positions):
            combined_img.paste(quadrant[i], pos)
            
        combined_frames.append(combined_img)
    
    # Create and save the video
    create_mp4_from_pil_images(combined_frames, output_path, song, fps)