import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Local AI Agent"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama-3-8b-instruct"  # Change to your registered GGUF model name
    DATABASE_PATH: str = os.path.join(os.path.dirname(__file__), "../../data/agent_memory.db")
    ALLOWED_ORIGINS: list = ["http://localhost:5173"]

    class Config:
        env_file = ".env"

settings = Settings()