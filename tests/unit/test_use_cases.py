"""Unit tests for AnswerLegalQAUseCase and citation verification."""

from unittest.mock import MagicMock

from src.application.dtos import LegalQARequest
from src.application.use_cases.answer_legal_qa import AnswerLegalQAUseCase
from src.domain.entities import Language, LegalChunk, LegalMetadata, RetrievedDocument


def test_answer_legal_qa_with_verified_citation():
    """QA use case should verify citations against retrieved articles."""
    mock_retriever = MagicMock()
    mock_llm = MagicMock()

    chunk315 = LegalChunk(
        chunk_id="chk_315",
        content="Article 315. A contract is formed when an offer and acceptance agree.",
        content_with_context="[Civil Code 2007] Article 315",
        metadata=LegalMetadata(
            law_name="Civil Code 2007",
            article_number=315,
            language=Language.ENGLISH,
        ),
    )
    mock_retriever.execute.return_value = [RetrievedDocument(chunk=chunk315, rrf_score=0.95)]

    mock_llm.generate.return_value = (
        "Under the Cambodian Civil Code, a contract is formed upon mutual agreement (Civil Code 2007, Article 315)."
    )

    qa_use_case = AnswerLegalQAUseCase(retriever=mock_retriever, llm=mock_llm)
    req = LegalQARequest(question="How is a contract formed in Cambodia?")
    response = qa_use_case.execute(req)

    assert "Article 315" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0]["article_number"] == 315
    assert response.citations[0]["is_verified"] is True
    assert response.citations[0]["law_name"] == "Civil Code 2007"


def test_answer_legal_qa_unverified_citation_flagged():
    """If LLM cites an article not in context, is_verified should be False."""
    mock_retriever = MagicMock()
    mock_llm = MagicMock()

    # Context only contains Article 10
    chunk10 = LegalChunk(
        chunk_id="chk_10",
        content="Article 10 content...",
        content_with_context="[Civil Code 2007] Article 10",
        metadata=LegalMetadata(
            law_name="Civil Code 2007",
            article_number=10,
            language=Language.ENGLISH,
        ),
    )
    mock_retriever.execute.return_value = [RetrievedDocument(chunk=chunk10)]

    # Model hallucinates Article 999
    mock_llm.generate.return_value = "According to Article 999, parties must act fairly."

    qa_use_case = AnswerLegalQAUseCase(retriever=mock_retriever, llm=mock_llm)
    req = LegalQARequest(question="What does Article 999 say?")
    response = qa_use_case.execute(req)

    assert len(response.citations) == 1
    assert response.citations[0]["article_number"] == 999
    assert response.citations[0]["is_verified"] is False
    assert response.citations[0]["has_factual_support"] is False


def test_compound_law_citation_distinguishes_civil_and_criminal():
    """Verify that compound law-and-article matching distinguishes Civil vs Criminal overlapping numbers."""
    mock_retriever = MagicMock()
    mock_llm = MagicMock()

    # Context contains Criminal Code Article 100
    chunk_crim100 = LegalChunk(
        chunk_id="chk_crim_100",
        content="មាត្រា ១០០.- បទល្មើសព្រហ្មទណ្ឌ...",
        content_with_context="[Criminal Code 2009] Article 100",
        metadata=LegalMetadata(
            law_name="Criminal Code 2009",
            article_number=100,
            language=Language.KHMER,
        ),
    )
    mock_retriever.execute.return_value = [RetrievedDocument(chunk=chunk_crim100)]

    # Model incorrectly cites Civil Code 2007 Article 100, but correctly cites Criminal Code Article 100
    mock_llm.generate.return_value = (
        "យោងតាម ក្រមរដ្ឋប្បវេណី មាត្រា ១០០ និង ក្រមព្រហ្មទណ្ឌ មាត្រា ១០០..."
    )

    qa_use_case = AnswerLegalQAUseCase(retriever=mock_retriever, llm=mock_llm)
    req = LegalQARequest(question="តើមាត្រា ១០០ ចែងអំពីអ្វី?")
    response = qa_use_case.execute(req)

    assert len(response.citations) == 2
    by_law = {c["law_name"]: c for c in response.citations}

    # Civil Code 100 was NOT retrieved, so it must be unverified
    assert by_law["Civil Code 2007"]["is_verified"] is False
    assert by_law["Civil Code 2007"]["has_factual_support"] is False

    # Criminal Code 100 WAS retrieved, so it must be verified
    assert by_law["Criminal Code 2009"]["is_verified"] is True
    assert by_law["Criminal Code 2009"]["has_factual_support"] is True


def test_citation_presence_distinguished_from_factual_support():
    """When model disclaims coverage, retrieved citations have presence verified but factual support False."""
    mock_retriever = MagicMock()
    mock_llm = MagicMock()

    chunk388 = LegalChunk(
        chunk_id="chk_388",
        content="មាត្រា ៣៨៨.- ការអនុវត្តតាមបញ្ញត្តិស្តីពីការសង...",
        content_with_context="[Civil Code 2007] Article 388",
        metadata=LegalMetadata(
            law_name="Civil Code 2007",
            article_number=388,
            language=Language.KHMER,
        ),
    )
    mock_retriever.execute.return_value = [RetrievedDocument(chunk=chunk388)]

    # Answer contains disclaimer that context does not address the question
    mock_llm.generate.return_value = (
        "អត្ថបទច្បាប់ដែលបានផ្ដល់ឱ្យមិនបានបញ្ជាក់អំពីសំណួរនេះទេ។ "
        "មាត្រា ៣៨៨ នៃក្រមរដ្ឋប្បវេណី ចែងអំពីការអនុវត្តតាមបញ្ញត្តិស្តីពីការសង មិនមែនការបង្កើតកិច្ចសន្យាទេ។"
    )

    qa_use_case = AnswerLegalQAUseCase(retriever=mock_retriever, llm=mock_llm)
    req = LegalQARequest(question="តើកិច្ចសន្យាបង្កើតឡើងដោយរបៀបណា?")
    response = qa_use_case.execute(req)

    assert len(response.citations) == 1
    cit = response.citations[0]
    assert cit["article_number"] == 388
    assert cit["is_verified"] is True  # Citation presence verified in context
    assert cit["has_factual_support"] is False  # Explicitly disclaimed as non-supportive
    assert "insufficient to address the question" in cit["support_notes"]

