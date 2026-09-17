"""
Hyperparameter tuning and training pipeline for Approach A3 (XLM-R Full Fine-Tuning).

Executes 6-configuration grid search:
  LR in {1e-5, 2e-5, 3e-5} x Weight Decay in {0.0, 0.01}

Features:
- Full end-to-end backpropagation across all 278M transformer parameters.
- AdamW optimizer with 10% linear warmup and Cosine Annealing.
- Automatic mixed-precision training (fp16 on CUDA / T4 GPU).
- Per-epoch validation loss and val MRR@10 tracking for model selection.
- Truncation rate measurement on training passages at max_length=256.
- Checkpoint management (best.pt and last.pt) with atomic saves.
- Comprehensive telemetry and benchmark evaluation on T-Q and T-T.
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
import time
from typing import Any, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.config.logging import get_logger, setup_logging
from src.dl.checkpoint import load_checkpoint, save_checkpoint
from src.dl.data import CorpusDataset, LegalPairsDataset, collate_pairs
from src.dl.evaluate import DenseRetriever, EvaluationHarness
from src.dl.losses import InfoNCELoss
from src.dl.metrics import evaluate_rankings
from src.dl.models.xlmr_finetune import XLMRFullFinetuneRetriever
from src.dl.seed import get_device, set_seed

setup_logging()
logger = get_logger(__name__)


def train_epoch(
    model: XLMRFullFinetuneRetriever,
    train_loader: DataLoader,
    optimizer: AdamW,
    scheduler: Any,
    loss_fn: InfoNCELoss,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    max_grad_norm: float = 1.0,
    dry_run: bool = False,
) -> float:
    """Run one fine-tuning epoch with mixed precision if available."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    use_amp = scaler is not None and device.type == "cuda"

    for i, batch in enumerate(train_loader):
        if dry_run and i >= 2:
            break

        queries = [f"query: {q}" for q in batch["queries"]]
        passages = [f"passage: {p}" for p in batch["passages"]]

        encoded_q = model.tokenizer(
            queries,
            max_length=model.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        encoded_p = model.tokenizer(
            passages,
            max_length=model.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                q_emb = model(encoded_q["input_ids"], encoded_q["attention_mask"])
                p_emb = model(encoded_p["input_ids"], encoded_p["attention_mask"])
                loss = loss_fn(q_emb, p_emb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            q_emb = model(encoded_q["input_ids"], encoded_q["attention_mask"])
            p_emb = model(encoded_p["input_ids"], encoded_p["attention_mask"])
            loss = loss_fn(q_emb, p_emb)

            loss.backward()
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate_val(
    model: XLMRFullFinetuneRetriever,
    val_dataset: LegalPairsDataset,
    loss_fn: InfoNCELoss,
    device: torch.device,
    batch_size: int = 32,
    dry_run: bool = False,
) -> tuple[float, float]:
    """Compute validation loss and validation MRR@10."""
    model.eval()
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_pairs,
    )

    total_loss = 0.0
    num_batches = 0
    all_q_emb: list[torch.Tensor] = []
    all_p_emb: list[torch.Tensor] = []

    for i, batch in enumerate(val_loader):
        if dry_run and i >= 2:
            break

        queries = [f"query: {q}" for q in batch["queries"]]
        passages = [f"passage: {p}" for p in batch["passages"]]

        encoded_q = model.tokenizer(
            queries,
            max_length=model.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        encoded_p = model.tokenizer(
            passages,
            max_length=model.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        q_emb = model(encoded_q["input_ids"], encoded_q["attention_mask"])
        p_emb = model(encoded_p["input_ids"], encoded_p["attention_mask"])

        loss = loss_fn(q_emb, p_emb)
        total_loss += loss.item()
        num_batches += 1

        all_q_emb.append(q_emb.cpu())
        all_p_emb.append(p_emb.cpu())

    val_loss = total_loss / max(1, num_batches)

    # Compute validation MRR@10
    all_q = torch.cat(all_q_emb, dim=0)
    all_p = torch.cat(all_p_emb, dim=0)
    sim = torch.matmul(all_q, all_p.t()).numpy()

    val_doc_ids = [r["doc_id"] for r in val_dataset.records[: len(all_q)]]
    top_k = min(10, len(val_doc_ids))
    rankings: list[list[str]] = []
    ground_truth: list[set[str]] = []

    for idx in range(len(val_doc_ids)):
        scores = sim[idx]
        top_indices = scores.argsort()[::-1][:top_k]
        rankings.append([val_doc_ids[j] for j in top_indices])
        ground_truth.append({val_doc_ids[idx]})

    metrics = evaluate_rankings(rankings, ground_truth, n_bootstrap=100)
    val_mrr10 = metrics.get("MRR@10", {}).get("mean", 0.0)

    return val_loss, val_mrr10


def run_single_config(
    run_name: str,
    lr: float,
    weight_decay: float,
    train_dataset: LegalPairsDataset,
    val_dataset: LegalPairsDataset,
    results_dir: Path,
    epochs: int = 5,
    patience: int = 3,
    batch_size: int = 16,
    temperature: float = 0.05,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    device: Optional[torch.device] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train one XLM-R full fine-tuning configuration."""
    set_seed(seed)
    target_device = device or get_device()

    checkpoints_dir = results_dir / "checkpoints" / run_name
    logs_dir = results_dir / "logs"
    metrics_dir = results_dir / "metrics"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_log_path = logs_dir / f"{run_name}.csv"

    model = XLMRFullFinetuneRetriever(device=target_device)
    loss_fn = InfoNCELoss(temperature=temperature, symmetric=True, normalize_embeddings=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_pairs,
    )

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    if warmup_steps > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    scaler = torch.amp.GradScaler("cuda") if target_device.type == "cuda" else None

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    history: list[dict[str, Any]] = []
    best_val_mrr = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    start_time = time.time()

    logger.info(
        f"Training {run_name} (LR={lr:.1e}, WD={weight_decay}) for {epochs} epochs on {target_device}..."
    )

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=target_device,
            scaler=scaler,
            max_grad_norm=max_grad_norm,
            dry_run=dry_run,
        )

        val_loss, val_mrr10 = evaluate_val(
            model=model,
            val_dataset=val_dataset,
            loss_fn=loss_fn,
            device=target_device,
            batch_size=batch_size,
            dry_run=dry_run,
        )

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_mrr10": round(val_mrr10, 4),
            "learning_rate": current_lr,
            "time_seconds": round(epoch_time, 2),
        })

        logger.info(
            f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val MRR@10: {val_mrr10:.4f} | Time: {epoch_time:.2f}s"
        )

        save_checkpoint(
            checkpoints_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_mrr=best_val_mrr,
        )

        if val_mrr10 > best_val_mrr:
            best_val_mrr = val_mrr10
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(
                checkpoints_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_mrr=val_mrr10,
            )
            logger.info(f"New best model saved with Val MRR@10: {val_mrr10:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping after {patience} epochs without improvement.")
                break

    total_time = time.time() - start_time

    # Write CSV history log
    with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_mrr10", "learning_rate", "time_seconds"],
        )
        writer.writeheader()
        writer.writerows(history)

    # Write telemetry JSON
    telemetry = {
        "run_name": run_name,
        "approach": "A3",
        "model_type": "xlmr_full_finetune",
        "lr": lr,
        "weight_decay": weight_decay,
        "trainable_parameters": trainable_params,
        "best_epoch": best_epoch,
        "best_val_mrr10": best_val_mrr,
        "total_wall_clock_seconds": round(total_time, 2),
        "device": str(target_device),
        "history": history,
    }
    with open(metrics_dir / f"{run_name}_train.json", "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    return telemetry


def run_tuning(
    splits_dir: Path = Path("data/05_splits"),
    results_dir: Path = Path("results"),
    epochs: int = 5,
    patience: int = 3,
    batch_size: int = 16,
    seed: int = 42,
    dry_run: bool = False,
    single_config: bool = False,
) -> Path:
    """Execute A3 hyperparameter tuning grid search and evaluate best model."""
    set_seed(seed)
    device = get_device()

    train_path = splits_dir / "train.jsonl"
    val_path = splits_dir / "val.jsonl"
    corpus_path = splits_dir / "corpus.jsonl"

    train_dataset = LegalPairsDataset(train_path)
    val_dataset = LegalPairsDataset(val_path)

    # 1. Measure and log sequence truncation rate at max_length=256
    probe_model = XLMRFullFinetuneRetriever(device=device)
    train_passages = [r["passage"] for r in train_dataset.records]
    trunc_rate = probe_model.compute_truncation_rate(train_passages, max_length=256)
    logger.info(f"Training passages truncation rate at max_length=256: {trunc_rate * 100:.2f}%")

    # 2. Grid search: LR in {1e-5, 2e-5, 3e-5} x WD in {0.0, 0.01}
    if single_config or dry_run:
        grid = [{"lr": 2e-5, "weight_decay": 0.01}]
    else:
        grid = [
            {"lr": 1e-5, "weight_decay": 0.0},
            {"lr": 1e-5, "weight_decay": 0.01},
            {"lr": 2e-5, "weight_decay": 0.0},
            {"lr": 2e-5, "weight_decay": 0.01},
            {"lr": 3e-5, "weight_decay": 0.0},
            {"lr": 3e-5, "weight_decay": 0.01},
        ]

    tuning_dir = results_dir / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    tuning_csv = tuning_dir / "a3.csv"

    tuning_rows: list[dict[str, Any]] = []
    logger.info(f"Starting A3 Full Fine-Tuning grid search ({len(grid)} configurations)...")

    for i, params in enumerate(grid, 1):
        lr = params["lr"]
        wd = params["weight_decay"]
        run_name = f"a3_lr{lr:.0e}_wd{wd}".replace("-0", "-")

        logger.info(f"\n[{i}/{len(grid)}] Running configuration: {run_name} (LR={lr:.1e}, WD={wd})")

        telemetry = run_single_config(
            run_name=run_name,
            lr=lr,
            weight_decay=wd,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            results_dir=results_dir,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            temperature=0.05,
            seed=seed,
            device=device,
            dry_run=dry_run,
        )

        best_epoch_info = max(telemetry["history"], key=lambda h: h["val_mrr10"])
        tuning_rows.append({
            "run_name": run_name,
            "lr": lr,
            "weight_decay": wd,
            "trainable_params": telemetry["trainable_parameters"],
            "best_epoch": best_epoch_info["epoch"],
            "best_val_mrr10": telemetry["best_val_mrr10"],
            "val_loss": round(best_epoch_info["val_loss"], 4),
            "train_loss": round(best_epoch_info["train_loss"], 4),
            "train_time_sec": telemetry["total_wall_clock_seconds"],
            "checkpoint_path": str(results_dir / "checkpoints" / run_name / "best.pt"),
        })

    # Save summary table
    df_tuning = pd.DataFrame(tuning_rows)
    df_tuning.sort_values(by="best_val_mrr10", ascending=False, inplace=True)
    df_tuning.to_csv(tuning_csv, index=False)
    logger.info(f"Saved A3 tuning summary table -> {tuning_csv}")

    print("\n" + "=" * 80)
    print("A3 XLM-R FULL FINE-TUNING SUMMARY")
    print("=" * 80)
    print(df_tuning.to_string(index=False))
    print("=" * 80 + "\n")

    # 3. Select winning model
    best_config = df_tuning.iloc[0]
    best_ckpt_src = Path(best_config["checkpoint_path"])
    best_ckpt_dest = results_dir / "checkpoints" / "a3_xlmr_finetune" / "best.pt"
    best_ckpt_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_ckpt_src, best_ckpt_dest)

    logger.info(
        f"Selected winning A3 model: '{best_config['run_name']}' "
        f"(Val MRR@10: {best_config['best_val_mrr10']:.4f}) -> {best_ckpt_dest}"
    )

    # 4. Evaluate winning model once on full corpus benchmarks
    logger.info("Evaluating selected A3 model on full benchmarks (T-Q and T-T)...")
    harness = EvaluationHarness(
        corpus_path=corpus_path,
        qa_path=splits_dir.parent.parent / "tests" / "evaluation" / "ground_truth_qa_kh.json",
        test_titles_path=splits_dir / "test_titles.jsonl",
        seed=seed,
    )

    best_model = XLMRFullFinetuneRetriever(device=device)
    load_checkpoint(best_ckpt_dest, model=best_model, device=device)

    retriever = DenseRetriever(best_model, harness.corpus, device=device, batch_size=batch_size)
    harness.run_full_evaluation(retriever, run_name="a3_xlmr", output_dir=results_dir / "metrics")

    return best_ckpt_dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune and evaluate Approach A3 XLM-R full fine-tuning.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs per configuration.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience.")
    parser.add_argument("--dry-run", action="store_true", help="Quick dry run on 2 batches.")
    parser.add_argument("--single-config", action="store_true", help="Run only the recommended single config.")
    args = parser.parse_args()

    run_tuning(
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        dry_run=args.dry_run,
        single_config=args.single_config,
    )
