from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path

class Settings(BaseSettings):
    # OpenAI Settings
    BASE_URL: str = "https://api.openai.com"
    API_KEY: str
    MODEL_NAME: str = "casperhansen/llama-3.3-70b-instruct-awq"
    QWEN_MODEL: str = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"

    # File Storage Settings
    UPLOAD_DIR: str = str(Path(__file__).parent / "data" / "raw")

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings():
    return Settings()
