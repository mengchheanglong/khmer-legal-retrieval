"""
Vocabulary management and text encoding for BiLSTM (A1) model.

Builds a vocabulary exclusively from the training split (train.jsonl)
to ensure zero test leakage. Handles special tokens (<pad>, <unk>),
frequency filtering, padding, and batched tensor conversion.
"""

from collections import Counter
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from src.config.logging import get_logger, setup_logging
from src.dl.text import normalize_khmer_text, tokenize_khmer

setup_logging()
logger = get_logger(__name__)

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


class KhmerVocab:
    """Vocabulary mapping Khmer word tokens to integers."""

    def __init__(
        self,
        token_to_id: Optional[dict[str, int]] = None,
    ) -> None:
        """Initialize vocabulary.

        Args:
            token_to_id: Optional existing token-to-index mapping.
        """
        if token_to_id is not None:
            self.token_to_id = token_to_id
            self.id_to_token = {idx: token for token, idx in token_to_id.items()}
        else:
            self.token_to_id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
            self.id_to_token = {PAD_ID: PAD_TOKEN, UNK_ID: UNK_TOKEN}

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    def token_id(self, token: str) -> int:
        """Lookup token ID with fallback to UNK_ID."""
        return self.token_to_id.get(token, UNK_ID)

    @classmethod
    def build_from_train_file(
        cls,
        train_jsonl_path: Path | str,
        min_freq: int = 1,
    ) -> "KhmerVocab":
        """Build vocabulary strictly from training pairs JSONL file.

        Args:
            train_jsonl_path: Path to data/05_splits/train.jsonl.
            min_freq: Minimum token occurrence frequency to include (default: 1).

        Returns:
            Constructed KhmerVocab instance.
        """
        path = Path(train_jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Training split file not found: {path}")

        logger.info(f"Building vocabulary from {path} (min_freq={min_freq})...")
        counts: Counter[str] = Counter()

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                q_tokens = tokenize_khmer(item["query"])
                p_tokens = tokenize_khmer(item["passage"])
                counts.update(q_tokens)
                counts.update(p_tokens)

        vocab = cls()

        # Sort deterministically: highest frequency first, then alphabetical
        sorted_tokens = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

        for token, count in sorted_tokens:
            if count >= min_freq and token not in vocab.token_to_id:
                new_idx = len(vocab.token_to_id)
                vocab.token_to_id[token] = new_idx
                vocab.id_to_token[new_idx] = token

        logger.info(
            f"Vocabulary built: {len(vocab)} tokens "
            f"({len(sorted_tokens)} total distinct tokens processed)."
        )
        return vocab

    def encode(
        self,
        text: str,
        max_seq_len: int = 256,
    ) -> tuple[list[int], list[int]]:
        """Encode a single text string into padded token IDs and an attention mask.

        Args:
            text: Khmer text string.
            max_seq_len: Maximum sequence length.

        Returns:
            Tuple of (token_ids_list, attention_mask_list).
        """
        tokens = tokenize_khmer(text)
        token_ids = [self.token_id(t) for t in tokens[:max_seq_len]]
        actual_len = len(token_ids)

        # Pad to max_seq_len
        pad_len = max(0, max_seq_len - actual_len)
        padded_ids = token_ids + [PAD_ID] * pad_len
        attention_mask = [1] * actual_len + [0] * pad_len

        return padded_ids, attention_mask

    def batch_encode(
        self,
        texts: Sequence[str],
        max_seq_len: int = 256,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of texts into PyTorch tensors.

        Args:
            texts: List of Khmer text strings.
            max_seq_len: Maximum sequence length.
            device: Target torch device.

        Returns:
            Tuple of (input_ids_tensor, attention_mask_tensor) of shape (B, max_seq_len).
        """
        all_ids: list[list[int]] = []
        all_masks: list[list[int]] = []

        for text in texts:
            ids, mask = self.encode(text, max_seq_len=max_seq_len)
            all_ids.append(ids)
            all_masks.append(mask)

        input_ids_tensor = torch.tensor(all_ids, dtype=torch.long, device=device)
        attention_mask_tensor = torch.tensor(all_masks, dtype=torch.float32, device=device)

        return input_ids_tensor, attention_mask_tensor

    def save(self, path: Path | str) -> None:
        """Save vocabulary to a JSON file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved vocabulary ({len(self)} tokens) -> {save_path}")

    @classmethod
    def load(cls, path: Path | str) -> "KhmerVocab":
        """Load vocabulary from a JSON file."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {load_path}")
        with open(load_path, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        vocab = cls(token_to_id=token_to_id)
        logger.info(f"Loaded vocabulary ({len(vocab)} tokens) from {load_path}")
        return vocab
