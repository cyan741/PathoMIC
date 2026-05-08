#!/usr/bin/env python3
"""
AMP MIC Regression Inference Script

Usage:
    python infer.py \
        --model_path /NAS/luyq/PLM_AMP_Regression/ckp/esm2-8M_lr5e-06_bs64_ep28_val_best.pth \
        --test_csv /NAS/luyq/AMP_datasets/splits1/test.csv \
        --output_dir /home/luyq/PLM_AMP_Regression/test \
        --batch_size 32 \
        --device 0

Output:
    {output_dir}/{model_name}_test_results.csv
    Original CSV columns + predict_MIC column (log10 scale)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Add scripts directory to sys.path so we can import project modules
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts')
sys.path.insert(0, SCRIPTS_DIR)

from plm_models import ESM2, load_tokenizer
from data_loader import seq2token


def parse_model_info(model_path: str):
    """
    Parse PLM type and ESM size from checkpoint filename.
    
    Example filename: esm2-8M_lr5e-06_bs64_ep28_val_best.pth
    Returns: plm_type='esm2-8M', esm_size='8M', model_name='esm2-8M_lr5e-06_bs64_ep28_val_best'
    """
    basename = os.path.basename(model_path)
    model_name = os.path.splitext(basename)[0]   # strip .pth
    plm_type = model_name.split('_')[0]           # e.g. 'esm2-8M'
    esm_size = plm_type.split('-')[-1]            # e.g. '8M'
    return plm_type, esm_size, model_name


def run_inference(model_path: str,
                  test_csv: str,
                  output_dir: str,
                  batch_size: int = 32,
                  device_id: str = '0',
                  max_length: int = 70):
    """
    Run inference, add predict_MIC column to original CSV, save results.
    """
    plm_type, esm_size, model_name = parse_model_info(model_path)
    print(f"[Model Info] PLM type: {plm_type} | ESM size: {esm_size}")
    print(f"[Model Name] {model_name}")

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device}")

    # --- Load test CSV ---
    df = pd.read_csv(test_csv)
    print(f"[Data] Loaded {len(df)} samples from {test_csv}")

    # --- Load tokenizer ---
    print("[Tokenizer] Loading...")
    tokenizer = load_tokenizer(plm_type)

    # --- Load model ---
    print("[Model] Loading checkpoint...")
    model = ESM2(
        plm_output='mean',
        head_type='3MLP',
        finetune_plm=True,
        esm_size=esm_size
    )
    checkpoint = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if missing:
        print(f"[Model] Missing keys (version mismatch, non-critical): {missing}")
    if unexpected:
        print(f"[Model] Unexpected keys: {unexpected}")
    model = model.to(device)
    model.eval()
    print("[Model] Checkpoint loaded.")

    # --- Inference ---
    sequences = df['Sequence'].tolist()
    all_preds = []

    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), batch_size), desc="Inference"):
            batch_seqs = sequences[start: start + batch_size]
            # Pad to max_length (same as MIC_Dataset.__getitem__)
            batch_seqs_padded = [seq.ljust(max_length, 'X') for seq in batch_seqs]
            input_ids = seq2token(batch_seqs_padded, tokenizer, device)
            outputs = model(input_ids)                   # shape: [batch, 1]
            preds = outputs.squeeze(-1).cpu().numpy()    # shape: [batch]
            all_preds.extend(preds.tolist())

    # --- Save results ---
    df['predict_MIC'] = all_preds
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_test_results.csv")
    df.to_csv(output_path, index=False)
    print(f"[Done] Saved {len(df)} rows -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained ESM2 MIC regression model."
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to .pth checkpoint, e.g. /NAS/luyq/PLM_AMP_Regression/ckp/esm2-8M_lr5e-06_bs64_ep28_val_best.pth'
    )
    parser.add_argument(
        '--test_csv', type=str, required=True,
        help='Path to test CSV with Sequence and Median_MIC columns'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='/home/luyq/PLM_AMP_Regression/test',
        help='Directory to save result CSV'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--device', type=str, default='0',
        help='CUDA device id (e.g. 0, 1) or "cpu"'
    )
    parser.add_argument(
        '--max_length', type=int, default=70,
        help='Sequence padding length (must match training setting)'
    )

    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_id=args.device,
        max_length=args.max_length,
    )


if __name__ == '__main__':
    main()
