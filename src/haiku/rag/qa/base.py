from haiku.rag.client import HaikuRAG
from haiku.rag.qa.prompts import SYSTEM_PROMPT


class QABase:
    _model: str = ""
    _system_prompt: str = SYSTEM_PROMPT

    def __init__(self, client: HaikuRAG, model: str = ""):
        self._model = model
        self._client = client

    async def answer(self, question: str) -> str:
        raise NotImplementedError(
            "QABase is an abstract class. Please implement the answer method in a subclass."
        )
