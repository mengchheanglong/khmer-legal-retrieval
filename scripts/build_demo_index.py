"""Build the demo index with the same article text as the evaluation harness."""
import hashlib
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def file_digest(path: Path) -> str:
    """Hash an input without loading the whole checkpoint into memory."""
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main() -> None:
    """Encode the fixed corpus using A3 and save provenance alongside the matrix."""
    from src.dl.data import CorpusDataset
    from src.dl.seed import set_seed
    from src.infrastructure.ai.torch_encoder_embedding import TorchEncoderEmbedding

    set_seed(42)
    torch.set_num_threads(4)
    checkpoint = ROOT / 'results/checkpoints/a3_xlmr_finetune/best.pt'
    corpus_path = ROOT / 'data/05_splits/corpus.jsonl'
    corpus = CorpusDataset(corpus_path)
    embedder = TorchEncoderEmbedding(checkpoint_path=checkpoint, device='cpu', batch_size=16)
    vectors = []
    texts = corpus.get_texts(use_context=False)
    for start in range(0, len(texts), 16):
        vectors.extend(embedder.embed(texts[start:start+16]))
        if start % 160 == 0:
            print(f'Encoded {min(start+16,len(texts))}/{len(texts)}', flush=True)
    output = ROOT / 'data/indices/khmer_benchmark_embeddings.pt'
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'embeddings': torch.tensor(vectors, dtype=torch.float32),
        'chunk_ids': [r['chunk_id'] for r in corpus.records],
        'checkpoint_sha256': file_digest(checkpoint),
        'corpus_sha256': file_digest(corpus_path),
        'preprocessing': 'CorpusDataset.get_texts(use_context=False)',
    }, output)
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
