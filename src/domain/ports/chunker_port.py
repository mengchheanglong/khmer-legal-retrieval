"""Abstract interface for legal text chunkers."""

from abc import ABC, abstractmethod

from src.domain.entities import LegalChunk


class ChunkerPort(ABC):
    """
    Port for splitting legal text into structured chunks.

    Implementations must chunk by legal hierarchy (Article-level)
    rather than arbitrary fixed-length splits.
    """

    @abstractmethod
    def chunk(
        self,
        text: str,
        law_name: str,
        language: str = "en",
        law_name_kh: str | None = None,
    ) -> list[LegalChunk]:
        """
        Split legal text into Article-level chunks with metadata.

        Args:
            text: Cleaned full text of a legal document.
            law_name: Name of the law for metadata tagging.
            language: Language code ("en" or "kh").
            law_name_kh: Khmer name of the law, used in Khmer chunk context.

        Returns:
            List of LegalChunk objects with hierarchical metadata.

        Raises:
            ChunkingError: If the text cannot be parsed into legal articles.
        """
        ...
