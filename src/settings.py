from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path

class Settings(BaseSettings):
    # OpenAI Settings
    BASE_URL: str
    API_KEY: str
    MODEL_NAME: str

    # File Storage Settings
    UPLOAD_DIR: str = str(Path(__file__).parent / "data" / "raw")

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings():
    return Settings()
