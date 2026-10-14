"""
DeepSeek Flash LLM Generation Adapter.

Implements LLMPort using DeepSeek's OpenAI-compatible API endpoint (https://api.deepseek.com).
Uses 'deepseek-chat' (DeepSeek Flash) for fast, cost-effective, cited legal question-answering.
"""

from typing import Optional

import openai
from openai import OpenAI
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from src.config.logging import get_logger
from src.config.settings import Settings, get_settings
from src.domain.entities import RetrievedDocument
from src.domain.exceptions import GenerationError
from src.domain.ports.llm_port import LLMPort
from src.infrastructure.ai.prompts import (
    LEGAL_QA_SYSTEM_PROMPT,
    LEGAL_QA_USER_TEMPLATE,
    format_context,
)

logger = get_logger(__name__)


class OpenAILLM(LLMPort):
    """
    Adapter for generating cited legal answers via DeepSeek Flash.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._settings = settings or get_settings()

        # Determine effective API key and sanitize
        raw_key = api_key or self._settings.deepseek_api_key or self._settings.openai_api_key
        self._api_key = raw_key.strip().strip('"').strip("'") if raw_key else ""

        # DeepSeek API endpoint
        self._base_url = base_url or self._settings.deepseek_base_url or "https://api.deepseek.com"
        self._model = model or self._settings.deepseek_model or "deepseek-chat"
        self._temperature = self._settings.llm_temperature

        if not self._api_key:
            logger.warning(
                "DEEPSEEK_API_KEY is not set. DeepSeekLLM will fail if generate() is called without a key."
            )

        self._client: Optional[OpenAI] = (
            OpenAI(api_key=self._api_key, base_url=self._base_url)
            if self._api_key
            else None
        )

    def generate(
        self,
        query: str,
        context_documents: list[RetrievedDocument],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        model_override: Optional[str] = None,
    ) -> str:
        """
        Generate a cited legal answer grounded strictly in the provided articles using DeepSeek Flash.

        Args:
            query: The user question.
            context_documents: Retrieved context articles.
            system_prompt: Optional system prompt override.
            temperature: Sampling temperature override.
            model_override: Optional model override.

        Returns:
            Generated answer text with article citations.
        """
        if not self._client:
            raise GenerationError(
                "DeepSeek client is not initialized. Please set DEEPSEEK_API_KEY in .env or the UI."
            )

        # Format context articles
        articles_data = [
            {
                "law_name": doc.chunk.metadata.law_name,
                "article_number": doc.chunk.metadata.article_number,
                "content": doc.chunk.content,
            }
            for doc in context_documents
        ]
        formatted_context = format_context(articles_data)

        user_content = LEGAL_QA_USER_TEMPLATE.format(
            question=query,
            context=formatted_context,
        )

        sys_prompt = system_prompt or LEGAL_QA_SYSTEM_PROMPT
        temp = temperature if temperature is not None else self._temperature
        model_name = model_override or self._model

        try:
            return self._call_chat_completion(sys_prompt, user_content, temp, model_name)
        except Exception as e:
            logger.error("DeepSeek generation failed", error=str(e), model=model_name)
            raise GenerationError(f"DeepSeek Flash generation error: {e}")

    @retry(
        retry=retry_if_not_exception_type((
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
            openai.BadRequestError,
        )),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
        reraise=True,
    )
    def _call_chat_completion(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float,
        model_name: str,
    ) -> str:
        """Execute DeepSeek Flash chat completion with retry."""
        response = self._client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content.strip()


# Alias for clean naming
DeepSeekLLM = OpenAILLM
