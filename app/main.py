from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import diffusion

app = FastAPI()

app.include_router(diffusion.router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8080, reload=True)
