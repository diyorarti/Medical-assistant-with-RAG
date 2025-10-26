from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
load_dotenv()

# project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    # loading envs
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # data loader configs 
    DATA_DIR:Path=PROJECT_ROOT / "data"

settings = Settings()