from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
import logging
import numpy as np
import cv2
import asyncio

# TODO: turbojpeg?
logger = logging.getLogger(__name__)
router = APIRouter()


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
async def stream_random_image(websocket:WebSocket):
    await websocket.accept()
    logger.info("Client Connected to Test Stream... hamba")

    try:
        frame_counter = 0
        target_fps = 30
        frame_time = 1.0 / target_fps  # 0.0333s per frame

        while True:
            loop_start = asyncio.get_event_loop().time()

            # Generate and send frame
            arr = np.random.randint(0,255,(512,512,3),dtype=np.uint8)
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
