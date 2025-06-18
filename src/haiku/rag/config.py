import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class AppConfig(BaseModel):
    ENV: str = "development"

    OLLAMA_BASE_URL: str = "http://localhost:11434"

    EMBEDDING_PROVIDER: str = "ollama"
    EMBEDDING_MODEL: str = "mxbai-embed-large"
    EMBEDDING_VECTOR_DIM: int = 1024
    CHUNK_SIZE: int = 256
    CHUNK_OVERLAP: int = 32


# Expose Config object for app to import
Config = AppConfig.model_validate(os.environ)
