"""
txt2imgBench - High-performance text-to-image benchmarking application

This is the main entry point for the refactored txt2imgBench application.
The application is now split into clean, focused components:
- DiffusionEngine: Handles all StreamDiffusion and PyTorch operations
- Api: Clean FastAPI backend for HTTP/WebSocket handling
- Config: Enhanced configuration with frontend parameter display
"""

import asyncio
import logging

from api import create_api
from config import Config

logger = logging.getLogger("uvicorn")


async def main():
    """
    Main entry point for the txt2imgBench application.
    """
    # Load configuration
    config = Config()

    # Log startup information
    logger.info("=" * 60)
    logger.info("Starting txt2imgBench - Text-to-Image Benchmark")
    logger.info("=" * 60)
    logger.info(f"Device: {config.device}")
    logger.info(f"Model: {config.base_model_id}")
    logger.info(f"LoRA: {config.hyper_sd_lora}")
    logger.info(f"TinyVAE: {config.use_tiny_vae}")
    logger.info(f"Profiling: {config.enable_profiling}")
    logger.info(f"Server: http://{config.host}:{config.port}")
    logger.info("=" * 60)

    # Create and initialize the API
    api = await create_api(config)

    # Run the server
    api.run()


if __name__ == "__main__":
    asyncio.run(main())