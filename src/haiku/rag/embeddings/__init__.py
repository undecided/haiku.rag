from haiku.rag.config import Config
from haiku.rag.embeddings.base import EmbedderBase
from haiku.rag.embeddings.ollama import Embedder as OllamaEmbedder


def get_embedder() -> EmbedderBase:
    """
    Factory function to get the appropriate embedder based on the configuration.
    """

    if Config.EMBEDDING_PROVIDER == "ollama":
        return OllamaEmbedder(Config.EMBEDDING_MODEL, Config.EMBEDDING_VECTOR_DIM)

    if Config.EMBEDDING_PROVIDER == "voyageai":
        try:
            from haiku.rag.embeddings.voyageai import Embedder as VoyageAIEmbedder
        except ImportError:
            raise ImportError(
                "VoyageAI embedder requires the 'voyageai' package. "
                "Please install haiku.rag with the 'voyageai' extra:"
                "uv pip install haiku.rag --extra voyageai"
            )
        return VoyageAIEmbedder(Config.EMBEDDING_MODEL, Config.EMBEDDING_VECTOR_DIM)
    raise ValueError(f"Unsupported embedding provider: {Config.EMBEDDING_PROVIDER}")
