"""
Use Case: Answer a legal question with cited articles.

Pipeline: Question → Cross-Lingual Query Expansion (if Khmer) → Hybrid Retrieve → Rerank → DeepSeek Generate → Verify Citations
"""

import re
from src.config.logging import get_logger
from src.application.dtos import LegalQARequest, LegalQAResponse, RetrievalRequest
from src.application.use_cases.hybrid_retrieve import HybridRetrieveUseCase
from src.domain.entities import Citation, RetrievedDocument
from src.domain.ports.llm_port import LLMPort

logger = get_logger(__name__)

# The system prompt enforcing strict citation behavior & multilingual response
LEGAL_SYSTEM_PROMPT = """You are an expert legal assistant specializing in Cambodian Civil and Commercial Law.

RULES:
1. Answer strictly based on the provided legal articles. Do NOT use outside knowledge.
2. Answer in the SAME language as the user's question (if the question is in Khmer, answer in fluent, formal Khmer; if in English, answer in English).
3. For every legal rule or statement you mention, cite the exact Law name, Chapter, and Article number.
   Format in English: (Civil Code 2007, Article XX)
   Format in Khmer: (ក្រមរដ្ឋប្បវេណី ឆ្នាំ២០០៧ មាត្រា XX)
4. If the provided context does not contain the answer, state:
   "The provided legal texts do not explicitly address this question." (or in Khmer: "អត្ថបទច្បាប់ដែលបានផ្ដល់ឱ្យមិនបានបញ្ជាក់អំពីសំណួរនេះទេ").
5. Be precise, thorough, and concise. Use legal terminology accurately.
6. When multiple articles are relevant, cite all of them.
7. Never fabricate or assume article numbers that are not in the context.
"""


class AnswerLegalQAUseCase:
    """
    Orchestrates the full legal Q&A pipeline.

    Supports cross-lingual queries (Khmer & English), retrieves relevant articles
    via hybrid search, generates cited answers via DeepSeek Flash, and verifies citations.
    """

    def __init__(
        self,
        retriever: HybridRetrieveUseCase,
        llm: LLMPort,
        native_khmer: bool = True,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._native_khmer = native_khmer

    def execute(self, request: LegalQARequest) -> LegalQAResponse:
        """
        Answer a legal question with article-level citations.

        Args:
            request: The legal question and retrieval parameters.

        Returns:
            LegalQAResponse with answer, reasoning, citations, and source articles.
        """
        logger.info("Processing legal question", model=request.model)

        # Step 1: Handle cross-lingual query translation if query is in Khmer
        search_query = self._prepare_search_query(request.question)

        # Step 2: Retrieve relevant articles
        retrieval_request = RetrievalRequest(
            query=search_query,
            top_k=request.top_k,
            law_filter=request.law_filter,
        )
        retrieved_docs = self._retriever.execute(retrieval_request)
        logger.info("Retrieved source articles", count=len(retrieved_docs))

        # Step 3: Generate answer with citations using DeepSeek Flash
        generate_kwargs = {
            "query": request.question,
            "context_documents": retrieved_docs,
            "system_prompt": LEGAL_SYSTEM_PROMPT,
        }
        if request.model and hasattr(self._llm, "generate"):
            try:
                answer = self._llm.generate(
                    query=request.question,
                    context_documents=retrieved_docs,
                    system_prompt=LEGAL_SYSTEM_PROMPT,
                    model_override=request.model,
                )
            except TypeError:
                answer = self._llm.generate(**generate_kwargs)
        else:
            answer = self._llm.generate(**generate_kwargs)

        reasoning_content = None
        raw_reasoning = getattr(self._llm, "last_reasoning_content", None)
        if isinstance(raw_reasoning, str):
            reasoning_content = raw_reasoning

        model_used = None
        if request.model:
            model_used = request.model
        else:
            raw_model = getattr(self._llm, "_model", None)
            if isinstance(raw_model, str):
                model_used = raw_model

        logger.info("Answer generated")

        # Step 4: Verify citations against retrieved context
        citations = self._verify_citations(answer, retrieved_docs)

        # Step 5: Build response
        source_articles = [
            {
                "law_name": doc.chunk.metadata.law_name,
                "article_number": doc.chunk.metadata.article_number,
                "chapter": doc.chunk.metadata.chapter,
                "content_preview": doc.chunk.content[:200],
                "relevance_score": doc.rerank_score or doc.rrf_score,
            }
            for doc in retrieved_docs
        ]

        return LegalQAResponse(
            question=request.question,
            answer=answer,
            reasoning_content=reasoning_content,
            model_used=model_used,
            citations=[c.model_dump() for c in citations],
            source_articles=source_articles,
        )

    def _prepare_search_query(self, question: str) -> str:
        """
        Prepare search query for retrieval.

        In native Khmer retrieval mode (default), the original query is passed directly
        to the fine-tuned Khmer dual encoder and Khmer-segmented BM25.
        If native_khmer is False, translates key legal terms to English search keywords.
        """
        if self._native_khmer:
            return question

        is_khmer = bool(re.search(r"[\u1780-\u17ff]", question))
        if not is_khmer:
            return question

        # Method 1: Try dynamic translation using DeepSeek Flash if client is active
        if hasattr(self._llm, "_client") and self._llm._client:
            try:
                prompt = (
                    "Translate this Cambodian legal question into 4-6 English legal search keywords "
                    "for searching statutory articles in the Cambodian Civil Code / Commercial Law.\n"
                    "Output ONLY the space-separated English keywords, nothing else.\n\n"
                    f"Question: {question}"
                )
                response = self._llm._client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=60,
                )
                translated = response.choices[0].message.content.strip()
                if translated and len(translated) > 2:
                    logger.info("DeepSeek translated query", search_query=translated)
                    return translated
            except Exception:
                pass

        # Method 2: Comprehensive legal dictionary fallback
        khmer_legal_mappings = {
            "កិច្ចសន្យា": "contract agreement",
            "បង្កើត": "formation create",
            "សំណើ": "offer",
            "ការព្រមទទួល": "acceptance",
            "ក្រមរដ្ឋប្បវេណី": "Civil Code",
            "មោឃៈ": "void invalid",
            "លុបចោល": "rescission terminate cancel",
            "សំណង": "damages compensation",
            "ខូចខាត": "loss harm defect",
            "កាតព្វកិច្ច": "obligation duty",
            "អាជ្ញាកណ្តាល": "arbitration",
            "សេចក្តីសម្រេច": "arbitral award",
            "ភតិសន្យា": "lease",
            "ជួល": "lease rent",
            "ទិញលក់": "sale purchase seller buyer",
            "តម្កល់": "deposit pledge mortgage",
            "អចលនវត្ថុ": "immovable property land",
            "ចលនវត្ថុ": "movable property",
            "អាយុកាលកំណត់": "extinctive prescription limitation period",
            "សុចរិត": "good faith",
            "បំណុល": "debt monetary obligation",
            "ទាមទារ": "claim demand remedy",
        }

        matched_terms = []
        for kh_term, en_term in khmer_legal_mappings.items():
            if kh_term in question:
                matched_terms.append(en_term)

        numbers = re.findall(r"\b\d+\b", question)
        if numbers:
            matched_terms.extend([f"Article {n}" for n in numbers])

        if matched_terms:
            search_query = " ".join(matched_terms)
            logger.info("Mapped Khmer query via dictionary", search_query=search_query)
            return search_query

        return question

    def _verify_citations(
        self,
        answer: str,
        retrieved_docs: list[RetrievedDocument],
    ) -> list[Citation]:
        """
        Extract and verify article citations from the generated answer.

        Supports both English ("Article 336") and Khmer ("មាត្រា ៣៣៦" or "មាត្រា 336") citations.

        Args:
            answer: The LLM-generated answer text.
            retrieved_docs: The articles used as context.

        Returns:
            List of Citation objects with verification status.
        """
        # Convert Khmer digits to Arabic digits (០-៩ -> 0-9)
        khmer_to_arabic = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")
        normalized_answer = answer.translate(khmer_to_arabic)

        # Match Article citations in English and Khmer
        english_articles = re.findall(r"Article\s+(\d+)", normalized_answer, re.IGNORECASE)
        khmer_articles = re.findall(r"មាត្រា\s*(\d+)", normalized_answer)
        all_cited_nums = set(int(n) for n in (english_articles + khmer_articles))

        # Build a lookup of retrieved article numbers
        retrieved_articles = {
            doc.chunk.metadata.article_number: doc for doc in retrieved_docs
        }

        citations: list[Citation] = []
        for article_num in all_cited_nums:
            is_verified = article_num in retrieved_articles

            excerpt = ""
            law_name = "Civil Code 2007"
            if is_verified:
                doc = retrieved_articles[article_num]
                excerpt = doc.chunk.content[:300]
                law_name = doc.chunk.metadata.law_name

            citations.append(
                Citation(
                    law_name=law_name,
                    article_number=article_num,
                    excerpt=excerpt,
                    is_verified=is_verified,
                )
            )

            if not is_verified:
                logger.warning("Unverified citation detected", article_number=article_num)

        return citations
