from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"  # backend/.env, regardless of CWD


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_prefix="PATHFINDER_", extra="ignore")

    # Bright Data: web identification of unknown objects (edge LLM grounds on this) +
    # training-data harvest for RSI.
    brightdata_api_key: str = ""
    brightdata_serp_zone: str = "serp"
    brightdata_unlocker_zone: str = "unlocker"

    # RunPod GPU endpoint — RSI training only (no LLM, no per-frame inference).
    runpod_api_key: str = ""
    rsi_trainer_url: str = ""

    # Collision thresholds (meters) served to the edge so tuning is centralized.
    collision_warn_m: float = 2.0
    collision_stop_m: float = 1.0

    # Edge model config. RSI publishes new detector weights to the manifest.
    llm_model: str = "Qwen2.5-0.5B-Instruct-q4f16_1-MLC"

    # RSI.
    rsi_enabled: bool = True
    rsi_min_confidence: float = 0.45
    rsi_review_every: int = 50
    rsi_data_dir: str = "data"

    host: str = "0.0.0.0"
    port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
