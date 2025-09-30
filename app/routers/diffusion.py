from fastapi import APIRouter
from fastapi.responses import Response

import numpy as np
import cv2

# TODO: turbojpeg?
router = APIRouter()


@router.get("/test/random")
async def getStream() -> Response:
    """
    Return random image
    """
    arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    return Response(content=buf.tobytes(), media_type="image/jpeg")
