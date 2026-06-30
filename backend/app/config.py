from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Backend = Literal["mock", "local", "runpod"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="PATHFINDER_", extra="ignore")

    # Where perception (YOLO + depth) runs. "mock" needs no GPU or keys.
    perception_backend: Backend = "mock"
    # Where Qwen runs. "runpod" points at a vLLM OpenAI-compatible endpoint.
    reasoning_backend: Backend = "mock"

    # RunPod-hosted Qwen (vLLM exposes an OpenAI-compatible API).
    runpod_qwen_base_url: str = "https://api.runpod.ai/v2/<endpoint-id>/openai/v1"
    runpod_qwen_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    runpod_api_key: str = ""

    # RunPod-hosted perception endpoint (serverless handler in runpod/perception).
    runpod_perception_url: str = ""

    # Bright Data: web search for unfamiliar objects + training-data gathering.
    brightdata_api_key: str = ""
    brightdata_serp_zone: str = "serp"
    brightdata_unlocker_zone: str = "unlocker"

    # Collision logic.
    collision_warn_m: float = 2.0
    collision_stop_m: float = 1.0
    # Real-world width (meters) used to turn pixel width into a coarse distance in mock/local fallback.
    focal_px: float = 700.0

    # Local model weights (used when *_backend == "local").
    yolo_weights: str = "yolov8n.pt"
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"

    # RSI loop.
    rsi_enabled: bool = True
    rsi_min_confidence: float = 0.45
    rsi_review_every: int = 50
    rsi_data_dir: str = "data"

    host: str = "0.0.0.0"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
