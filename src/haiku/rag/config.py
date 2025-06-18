import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from haiku.rag.utils import get_default_data_dir

load_dotenv()


class AppConfig(BaseModel):
    ENV: str = "development"

    DEFAULT_DATA_DIR: Path = get_default_data_dir()

    EMBEDDING_PROVIDER: str = "ollama"
    EMBEDDING_MODEL: str = "mxbai-embed-large"
    EMBEDDING_VECTOR_DIM: int = 1024

    CHUNK_SIZE: int = 256
    CHUNK_OVERLAP: int = 32

    OLLAMA_BASE_URL: str = "http://localhost:11434"


# Expose Config object for app to import
Config = AppConfig.model_validate(os.environ)
