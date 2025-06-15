class EmbedderBase:
    _model: str = ""
    _vector_dim: int = 0

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            "Embedder is an abstract class. Please implement the embed method in a subclass."
        )
