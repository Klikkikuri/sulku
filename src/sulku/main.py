from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

# import fasttext
from sulku.wpapi import wpapi_router


# Global dictionary to hold pre-loaded models in memory
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models ONCE on server startup
    try:
        # models["gemini"] = fasttext.load_model("models/gemini_detector.ftz")
        # models["deepseek"] = fasttext.load_model("models/deepseek_detector.ftz")
        # models["qwen"] = fasttext.load_model("models/qwen_detector.ftz")
        yield
    finally:
        # Clean up resources on shutdown if necessary
        models.clear()


def create_app() -> FastAPI:
    app = FastAPI(title="AI Text Classifier Service", lifespan=lifespan)

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    app.include_router(wpapi_router)

    return app
