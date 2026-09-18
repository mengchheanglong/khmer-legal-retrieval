# Model Checkpoints & External Hosting Guide

Per Section 9 of the course rubric, model checkpoints exceeding 50 MB must not be committed to git and are hosted externally.

---

## Checkpoint Inventory

| Approach | Model Directory | File | Size | Tracked in Git? | Description |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **A1** | `results/checkpoints/a1_bilstm/` | `best.pt` | ~23.6 MB | Excluded (`*.pt`) | BiLSTM dual encoder (random init + 1.97M trainable parameters) |
| **A2** | `results/checkpoints/a2_linear_probe/` | `best.pt` | ~1.1 GB | Excluded (`*.pt`) | XLM-R linear probe checkpoint (backbone + 768→768 head) |
| **A3** | `results/checkpoints/a3_xlmr_finetune/` | `best.pt` | ~3.3 GB | Excluded (`*.pt`) | **Winning model**: Fully fine-tuned XLM-RoBERTa (`multilingual-e5-base`) + optimizer & scheduler state |

---

## External Hosting (Hugging Face Hub)

The winning Approach A3 model weights can be published to Hugging Face Hub under:
`https://huggingface.co/mengchheanglong/khmer-legal-xlmr-retriever`

### 1. Uploading Checkpoint to Hugging Face Hub

Using the `huggingface_hub` Python library or CLI:

```bash
pip install huggingface_hub
huggingface-cli login
```

```python
from huggingface_hub import HfApi

api = HfApi()

# Create repository if not already created
api.create_repo(
    repo_id="mengchheanglong/khmer-legal-xlmr-retriever",
    repo_type="model",
    exist_ok=True,
)

# Upload the winning A3 checkpoint
api.upload_file(
    path_or_fileobj="results/checkpoints/a3_xlmr_finetune/best.pt",
    path_in_repo="best.pt",
    repo_id="mengchheanglong/khmer-legal-xlmr-retriever",
    repo_type="model",
)
```

---

### 2. Downloading Pre-Trained Weights

To run inference or evaluate without retraining from scratch:

```python
from huggingface_hub import hf_hub_download
import os

os.makedirs("results/checkpoints/a3_xlmr_finetune", exist_ok=True)

checkpoint_path = hf_hub_download(
    repo_id="mengchheanglong/khmer-legal-xlmr-retriever",
    filename="best.pt",
    local_dir="results/checkpoints/a3_xlmr_finetune",
)
print(f"Downloaded checkpoint to: {checkpoint_path}")
```

---

## Evaluation & Inference with Checkpoint

Once downloaded, evaluate the checkpoint directly with the evaluation harness:

```bash
python -m src.dl.evaluate --model a3_xlmr --split test_questions
```
