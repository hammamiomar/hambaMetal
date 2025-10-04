from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
import logging
import math
import numpy as np
import cv2
import asyncio
import torch
from typing import Optional

# import new refactored modules
from src.config import PipelineConfig
from src.audio import AudioAnalyzer, AudioFeatures
from src.diffusion import SD15Pipeline, StreamDiffusionPipeline
from src.latent import LatentGenerator
from src.prompt import PromptInterpolator
from src.profiling import get_profiler

# TODO: turbojpeg?
logger = logging.getLogger(__name__)
router = APIRouter()

# global pipeline state (initialized on first generation request)
_pipeline: Optional[StreamDiffusionPipeline] = None
_audio_analyzer: Optional[AudioAnalyzer] = None
_latent_gen: Optional[LatentGenerator] = None
_prompt_interp: Optional[PromptInterpolator] = None
_config: Optional[PipelineConfig] = None


def get_or_create_pipeline() -> StreamDiffusionPipeline:
    """lazy initialization of pipeline (slow first time)"""
    global _pipeline, _audio_analyzer, _latent_gen, _prompt_interp, _config

    if _pipeline is None:
        logger.info("initializing diffusion pipeline (this may take a minute)...")

        # use fast preset for real-time performance
        _config = PipelineConfig.fast_preset()

        # create components
        _audio_analyzer = AudioAnalyzer(_config.audio)
        _latent_gen = LatentGenerator(
            config=_config.latent,
            device=_config.diffusion.device,
            dtype=_config.diffusion.get_torch_dtype(),
        )
        _prompt_interp = PromptInterpolator(
            config=_config.prompt,
            device=_config.diffusion.device,
            dtype=_config.diffusion.get_torch_dtype(),
        )

        # create diffusion pipeline
        base_pipeline = SD15Pipeline(
            diffusion_config=_config.diffusion,
            inference_config=_config.inference,
        )

        # wrap with stream optimizations
        _pipeline = StreamDiffusionPipeline(
            base_pipeline=base_pipeline,
            stream_config=_config.stream,
        )

        logger.info("pipeline initialized successfully")

    return _pipeline


def tensor_to_buffer(tensor: torch.Tensor) -> bytes:
    """
    convert pytorch tensor (C, H, W) in range [0,1] to jpeg bytes
    optimized for streaming
    """
    # convert to numpy (H, W, C) uint8
    arr = tensor.cpu().permute(1, 2, 0).numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)

    # rgb to bgr for opencv
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # encode jpeg
    _, buf = cv2.imencode(".jpg", arr_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def array_to_buffer(arr: np.ndarray) -> bytes:
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)

    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", arr_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


@router.get("/test/random")
async def get_random_image() -> Response:
    """
    Return random image
    """
    arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    return Response(content=array_to_buffer(arr), media_type="image/jpeg")


@router.websocket("/ws/test/")
async def stream_random_image(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client Connected to Test Stream... hamba")

    try:
        frame_counter = 0
        target_fps = 30
        frame_time = 1.0 / target_fps  # 0.0333s per frame

        while True:
            loop_start = asyncio.get_event_loop().time()

            # Generate and send frame
            arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            frame_buffer = array_to_buffer(arr)
            await websocket.send_bytes(frame_buffer)
            frame_counter += 1

            # Sleep for remaining time to hit target FPS
            elapsed = asyncio.get_event_loop().time() - loop_start
            sleep_time = max(0, frame_time - elapsed)
            await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        logger.info("Client Disconnected from Test Stream... hamba")
    except Exception as e:
        logger.error(f"Error in Test Stream: {e}")


@router.websocket("/ws/generate/")
async def stream_music_visualization(websocket: WebSocket):
    """
    real-time music visualization using streamdiffusion-inspired pipeline
    generates frames on-the-fly synchronized to music
    """
    await websocket.accept()
    logger.info("client connected to music visualization stream")

    try:
        # get pipeline (lazy init)
        pipeline = get_or_create_pipeline()
        profiler = get_profiler()

        # reset state for new session
        pipeline.reset_state()
        _latent_gen.reset()
        _prompt_interp.reset()

        # for demo, use hardcoded audio path
        # TODO: receive audio file from client or use live audio
        # audio_path = "path/to/your/song.mp3"

        # DEMO MODE: generate simple latent walk without audio
        # this demonstrates the pipeline working in real-time
        logger.info("demo mode: generating without audio (simple latent walk)")

        # get latent shape
        latent_shape = pipeline.get_latent_shape()

        # initialize base noise
        torch.manual_seed(42)
        base_noise_x = torch.randn(latent_shape, device=_config.diffusion.device, dtype=_config.diffusion.get_torch_dtype())
        torch.manual_seed(43)
        base_noise_y = torch.randn(latent_shape, device=_config.diffusion.device, dtype=_config.diffusion.get_torch_dtype())

        # encode base prompts
        base_prompt = _config.prompt.base_prompt
        chroma_prompts = _config.prompt.chroma_prompts

        with profiler.profile("encode_prompts"):
            base_embeds = pipeline.encode_prompt(base_prompt)
            chroma_embeds = pipeline.encode_prompt(chroma_prompts)

        # streaming parameters
        target_fps = 5  # conservative for real-time (increase with better hardware)
        frame_time = 1.0 / target_fps
        num_demo_frames = 100  # demo length

        frame_counter = 0

        while frame_counter < num_demo_frames:
            loop_start = asyncio.get_event_loop().time()

            # compute angle for circular motion
            angle = (frame_counter / num_demo_frames) * 2 * math.pi * 3  # 3 full rotations
            energy = 0.5 + 0.5 * math.sin(angle * 2)  # simulated music energy

            # generate latent (convert angle to tensor for cos/sin)
            angle_tensor = torch.tensor(angle, device=_config.diffusion.device, dtype=_config.diffusion.get_torch_dtype())
            latent = (
                torch.cos(angle_tensor) * base_noise_x + torch.sin(angle_tensor) * base_noise_y
            )

            # simple prompt interpolation (slerp between base and first chroma)
            prompt_alpha = energy * 0.3
            if base_embeds.dim() == 3:
                current_embeds = base_embeds[0:1]
            else:
                current_embeds = base_embeds.unsqueeze(0)

            # generate frame with profiling
            with profiler.profile("generate_frame"):
                frame_tensor = pipeline.generate_frame_with_coherence(
                    latent=latent,
                    prompt_embeds=current_embeds,
                    music_energy=energy,
                )

            # convert to jpeg
            with profiler.profile("encode_jpeg"):
                frame_buffer = tensor_to_buffer(frame_tensor)

            # send to client
            await websocket.send_bytes(frame_buffer)
            frame_counter += 1

            # log progress
            if frame_counter % 10 == 0:
                avg_gen_time = profiler.get_avg_time("generate_frame")
                avg_jpeg_time = profiler.get_avg_time("encode_jpeg")
                logger.info(
                    f"frame {frame_counter}/{num_demo_frames} | "
                    f"gen: {avg_gen_time:.1f}ms | jpeg: {avg_jpeg_time:.1f}ms"
                )

            # sleep for remaining time
            elapsed = asyncio.get_event_loop().time() - loop_start
            sleep_time = max(0, frame_time - elapsed)
            await asyncio.sleep(sleep_time)

        # print profiling summary
        profiler.print_summary()
        if _config.profiling.export_json:
            profiler.export_json()

        logger.info("demo stream completed")

    except WebSocketDisconnect:
        logger.info("client disconnected from music visualization stream")
    except Exception as e:
        logger.error(f"error in music visualization stream: {e}", exc_info=True)
    finally:
        # cleanup
        if _pipeline is not None:
            _pipeline.cleanup()
