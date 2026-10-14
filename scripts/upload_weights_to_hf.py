"""Script to upload trained model weights to Hugging Face Hub."""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Upload trained weights to Hugging Face Hub")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--repo-id", default="mengchheanglong/khmer-legal-xlmr-retriever")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Error: Hugging Face token not provided. Pass --token or set HF_TOKEN.", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    user_info = api.whoami()
    print(f"Authenticated as: {user_info.get('name')}")
    print(f"Target repository: {args.repo_id}")

    base_dir = Path(__file__).resolve().parent.parent
    clean_ckpt = base_dir / "results" / "checkpoints" / "a3_xlmr_finetune" / "clean_model.pt"
    fp16_ckpt = base_dir / "results" / "checkpoints" / "a3_xlmr_finetune" / "a3_encoder_fp16.pt"

    if clean_ckpt.exists():
        size_mb = clean_ckpt.stat().st_size / (1024 * 1024)
        print(f"\n1/2 Uploading {clean_ckpt.name} ({size_mb:.1f} MB) -> 'best.pt' ...")
        api.upload_file(
            path_or_fileobj=str(clean_ckpt),
            path_in_repo="best.pt",
            repo_id=args.repo_id,
            repo_type="model",
        )
        print("Successfully uploaded 'best.pt'!")
    else:
        print(f"Warning: {clean_ckpt} not found!", file=sys.stderr)

    if fp16_ckpt.exists():
        size_mb = fp16_ckpt.stat().st_size / (1024 * 1024)
        print(f"\n2/2 Uploading {fp16_ckpt.name} ({size_mb:.1f} MB) -> 'a3_encoder_fp16.pt' ...")
        api.upload_file(
            path_or_fileobj=str(fp16_ckpt),
            path_in_repo="a3_encoder_fp16.pt",
            repo_id=args.repo_id,
            repo_type="model",
        )
        print("Successfully uploaded 'a3_encoder_fp16.pt'!")

    print("\n--- All uploads complete ---")
    files = api.list_repo_files(repo_id=args.repo_id)
    print("Files currently in repo:", files)


if __name__ == "__main__":
    main()
