import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any

import uvicorn
from config import Config
from diffusion_engine import DiffusionEngine
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger("uvicorn")


class PredictInputModel(BaseModel):
    """
    The input model for the /predict endpoint.
    """
    prompt: str


class PredictResponseModel(BaseModel):
    """
    The response model for the /predict endpoint.
    """
    base64_image: str
    fps: float
    inference_time_ms: float


class BenchmarkStatsModel(BaseModel):
    """
    The response model for benchmark statistics.
    """
    current_fps: float
    average_fps: float
    min_fps: float
    max_fps: float
    total_generations: int


class Api:
    """
    Clean FastAPI backend focused on HTTP/WebSocket handling.
    Uses DiffusionEngine as a service for all torch operations.
    """
    def __init__(self, config: Config):
        self.config = config

        # Initialize the diffusion engine
        self.diffusion_engine = DiffusionEngine(config)

        # Setup FastAPI
        self.app = FastAPI(
            title="txt2imgBench API",
            description="High-performance text-to-image benchmarking API",
            version="1.0.0"
        )
        self._setup_routes()

    async def initialize(self):
        """Initialize the API and diffusion engine."""
        logger.info("Initializing API...")
        await self.diffusion_engine.initialize()
        logger.info("API initialization complete")

    def _setup_routes(self):
        """
        Setup WebSocket and static file routes.
        """
        # WebSocket endpoint for real-time image generation
        self.app.add_websocket_route("/ws", self._websocket_endpoint)

        # REST endpoint for configuration
        self.app.add_api_route("/config", self._get_config, methods=["GET"])

        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Serve static files
        frontend_dist = Path(__file__).parent / "frontend" / "dist"
        if frontend_dist.exists():
            self.app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="public")

    async def _get_config(self) -> Dict[str, Any]:
        """
        REST endpoint to get current configuration for frontend display.
        """
        return self.config.to_dict()

    async def _websocket_endpoint(self, websocket: WebSocket):
        """
        WebSocket endpoint for real-time image generation and stats.
        """
        await websocket.accept()
        logger.info("WebSocket connection established")

        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)

                message_type = message.get("type")

                if message_type == "generate":
                    # Generate single image
                    prompt = message.get("prompt", "")
                    result = await self.diffusion_engine.generate_image(prompt)
                    await websocket.send_text(json.dumps({
                        "type": "image",
                        "data": result
                    }))

                elif message_type == "start_benchmark":
                    # Start continuous generation
                    prompt = message.get("prompt", "")
                    await self._start_benchmark(websocket, prompt)

                elif message_type == "get_stats":
                    # Send current stats
                    stats = self.diffusion_engine.get_stats_dict()
                    await websocket.send_text(json.dumps({
                        "type": "stats",
                        "data": stats
                    }))

                elif message_type == "get_config":
                    # Send current configuration
                    config_data = self.config.to_dict()
                    await websocket.send_text(json.dumps({
                        "type": "config",
                        "data": config_data
                    }))

        except WebSocketDisconnect:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket.close()

    async def _start_benchmark(self, websocket: WebSocket, prompt: str):
        """
        Start continuous benchmark mode using the diffusion engine.
        """
        logger.info("Starting benchmark mode")
        benchmark_active = True

        async def check_stop_signal():
            nonlocal benchmark_active
            try:
                while benchmark_active:
                    try:
                        # Non-blocking check for stop message
                        data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                        message = json.loads(data)
                        if message.get("type") == "stop_benchmark":
                            logger.info("Benchmark stopped by client")
                            benchmark_active = False
                            break
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        break
            except Exception as e:
                logger.error(f"Stop signal check error: {e}")
                benchmark_active = False

        # Start stop signal checker task
        stop_task = asyncio.create_task(check_stop_signal())

        try:
            # Use diffusion engine's async generator for image generation
            async for result in self.diffusion_engine.benchmark_generator(prompt):
                if not benchmark_active:
                    break

                # Send result via WebSocket
                await websocket.send_text(json.dumps({
                    "type": "benchmark_image",
                    "data": result
                }))

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected during benchmark")
        except Exception as e:
            logger.error(f"Benchmark error: {e}")
        finally:
            benchmark_active = False
            stop_task.cancel()
            try:
                await stop_task
            except asyncio.CancelledError:
                pass

    def run(self):
        """
        Run the API server.
        """
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            workers=self.config.workers,
        )


async def create_api(config: Config) -> Api:
    """
    Factory function to create and initialize the API.
    """
    api = Api(config)
    await api.initialize()
    return api