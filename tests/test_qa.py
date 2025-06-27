from typing import TYPE_CHECKING

import pytest
from datasets import Dataset

from haiku.rag.client import HaikuRAG
from haiku.rag.qa.ollama import QA

if TYPE_CHECKING:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent))
    from llm_judge import LLMJudge


@pytest.mark.asyncio
async def test_qa_with_dataset_question(qa_corpus: Dataset, llm_judge: "LLMJudge"):
    """Test QA with actual question from the dataset using LLM judge."""
    client = HaikuRAG(":memory:")
    qa = QA(client)

    # Use the first document from the corpus
    doc = qa_corpus[1]

    # Add the document to database
    await client.create_document(
        content=doc["document_extracted"], uri=doc["document_id"]
    )

    question = doc["question"]
    expected_answer = doc["answer"]

    answer = await qa.answer(question)
    # Use LLM judge to evaluate answer equivalence
    is_equivalent = await llm_judge.judge_answers(question, answer, expected_answer)

    assert isinstance(answer, str)
    assert len(answer) > 0
    assert is_equivalent, (
        f"Generated answer not equivalent to expected answer.\nQuestion: {question}\nGenerated: {answer}\nExpected: {expected_answer}"
    )
