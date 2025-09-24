from dataclasses import dataclass, field
from typing import List, Literal, Dict, Any
import torch
import os

@dataclass
class Config:
    """
    Configuration for txt2imgBench with MPS optimization focus.
    """

    ####################################################################
    # Server
    ####################################################################
    host: str = "127.0.0.1"
    port: int = 9091
    workers: int = 1

    ####################################################################
    # Model configuration for manual setup
    ####################################################################
    # Base model for Hyper-SD
    base_model_id: str = "runwayml/stable-diffusion-v1-5"

    # Hyper-SD LoRA configuration
    hyper_sd_repo: str = "ByteDance/Hyper-SD"
    hyper_sd_lora: str = "Hyper-SD15-1step-lora.safetensors"

    # Device and optimization settings for MPS
    device: torch.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype: torch.dtype = torch.float16

    # Image generation settings
    width: int = 512
    height: int = 512

    # Inference settings for 1-step generation
    num_inference_steps: int = 1
    guidance_scale: float = 0.0  # No CFG for 1-step
    eta: float = 1.0  # TCD scheduler parameter

    # StreamDiffusion settings
    t_index_list: List[int] = field(default_factory=lambda: [0])  # Single step
    frame_buffer_size: int = 1
    use_denoising_batch: bool = False  # Simpler for benchmarking
    cfg_type: Literal["none"] = "none"  # Required for txt2img

    # FPS measurement settings
    fps_window_size: int = 10  # Rolling average window
    warmup_iterations: int = 5  # Warmup before measuring

    # VAE optimization
    use_tiny_vae: bool = True
    vae_id: str = "madebyollin/taesd"  # TinyVAE for 4x faster encoding/decoding

    # Safety and optimization
    use_safety_checker: bool = False
    acceleration: Literal["none", "xformers"] = "none"  # MPS doesn't support TensorRT

    # Performance analysis settings
    enable_profiling: bool = True
    detailed_timing: bool = True
    memory_tracking: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format for frontend display.
        Organizes parameters into logical sections with descriptions.
        """
        return {
            "server": {
                "title": "Server Configuration",
                "description": "API server and network settings",
                "parameters": {
                    "host": {
                        "value": self.host,
                        "description": "Server host address",
                        "type": "string"
                    },
                    "port": {
                        "value": self.port,
                        "description": "Server port number",
                        "type": "integer"
                    },
                    "workers": {
                        "value": self.workers,
                        "description": "Number of worker processes",
                        "type": "integer"
                    }
                }
            },
            "model": {
                "title": "Model Configuration",
                "description": "Base model and LoRA settings",
                "parameters": {
                    "base_model_id": {
                        "value": self.base_model_id,
                        "description": "Base Stable Diffusion model",
                        "type": "string"
                    },
                    "hyper_sd_repo": {
                        "value": self.hyper_sd_repo,
                        "description": "Hyper-SD repository on HuggingFace",
                        "type": "string"
                    },
                    "hyper_sd_lora": {
                        "value": self.hyper_sd_lora,
                        "description": "Hyper-SD LoRA weights file",
                        "type": "string"
                    }
                }
            },
            "device": {
                "title": "Device & Optimization",
                "description": "Hardware acceleration and optimization settings",
                "parameters": {
                    "device": {
                        "value": str(self.device),
                        "description": "Compute device (MPS, CPU, CUDA)",
                        "type": "string"
                    },
                    "dtype": {
                        "value": str(self.dtype),
                        "description": "Tensor data type for memory efficiency",
                        "type": "string"
                    },
                    "acceleration": {
                        "value": self.acceleration,
                        "description": "Memory optimization method",
                        "type": "string"
                    }
                }
            },
            "image_generation": {
                "title": "Image Generation",
                "description": "Image output and inference settings",
                "parameters": {
                    "width": {
                        "value": self.width,
                        "description": "Generated image width in pixels",
                        "type": "integer"
                    },
                    "height": {
                        "value": self.height,
                        "description": "Generated image height in pixels",
                        "type": "integer"
                    },
                    "num_inference_steps": {
                        "value": self.num_inference_steps,
                        "description": "Number of denoising steps (1-step for speed)",
                        "type": "integer"
                    },
                    "guidance_scale": {
                        "value": self.guidance_scale,
                        "description": "Classifier-free guidance scale (0.0 = disabled)",
                        "type": "float"
                    },
                    "eta": {
                        "value": self.eta,
                        "description": "TCD scheduler parameter (noise scaling)",
                        "type": "float"
                    }
                }
            },
            "stream_diffusion": {
                "title": "StreamDiffusion Settings",
                "description": "Real-time diffusion pipeline configuration",
                "parameters": {
                    "t_index_list": {
                        "value": self.t_index_list,
                        "description": "Timestep indices for denoising",
                        "type": "list"
                    },
                    "frame_buffer_size": {
                        "value": self.frame_buffer_size,
                        "description": "Number of frames to buffer",
                        "type": "integer"
                    },
                    "use_denoising_batch": {
                        "value": self.use_denoising_batch,
                        "description": "Enable batched denoising operations",
                        "type": "boolean"
                    },
                    "cfg_type": {
                        "value": self.cfg_type,
                        "description": "Classifier-free guidance type",
                        "type": "string"
                    }
                }
            },
            "vae": {
                "title": "VAE Configuration",
                "description": "Variational Autoencoder settings for encoding/decoding",
                "parameters": {
                    "use_tiny_vae": {
                        "value": self.use_tiny_vae,
                        "description": "Use TinyVAE for 4x faster operations",
                        "type": "boolean"
                    },
                    "vae_id": {
                        "value": self.vae_id,
                        "description": "TinyVAE model identifier",
                        "type": "string"
                    }
                }
            },
            "performance": {
                "title": "Performance Analysis",
                "description": "Benchmarking and profiling settings",
                "parameters": {
                    "fps_window_size": {
                        "value": self.fps_window_size,
                        "description": "Rolling window size for FPS calculation",
                        "type": "integer"
                    },
                    "warmup_iterations": {
                        "value": self.warmup_iterations,
                        "description": "Number of warmup runs before measuring",
                        "type": "integer"
                    },
                    "enable_profiling": {
                        "value": self.enable_profiling,
                        "description": "Enable PyTorch profiler for detailed analysis",
                        "type": "boolean"
                    },
                    "detailed_timing": {
                        "value": self.detailed_timing,
                        "description": "Track detailed timing breakdowns",
                        "type": "boolean"
                    },
                    "memory_tracking": {
                        "value": self.memory_tracking,
                        "description": "Monitor memory usage during generation",
                        "type": "boolean"
                    }
                }
            },
            "safety": {
                "title": "Safety & Security",
                "description": "Content filtering and safety settings",
                "parameters": {
                    "use_safety_checker": {
                        "value": self.use_safety_checker,
                        "description": "Enable NSFW content filtering",
                        "type": "boolean"
                    }
                }
            },
            "system_info": {
                "title": "System Information",
                "description": "Runtime environment details",
                "parameters": {
                    "mps_available": {
                        "value": torch.backends.mps.is_available(),
                        "description": "Apple Silicon MPS acceleration available",
                        "type": "boolean"
                    },
                    "torch_version": {
                        "value": torch.__version__,
                        "description": "PyTorch version",
                        "type": "string"
                    }
                }
            }
        }