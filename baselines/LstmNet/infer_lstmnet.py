#!/usr/bin/env python3
"""Inference for LstmNet MIC regression checkpoints."""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i + 1 for i, a in enumerate(AA)}
AA_TO_IDX["X"] = 0
MAX_LEN = 32


class LstmNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.embedding = nn.Embedding(21, cfg["embedding_dim"], padding_idx=0)
        self.lstm = nn.LSTM(
            cfg["embedding_dim"],
            cfg["hidden_num"],
            cfg["num_layer"],
            batch_first=True,
            bidirectional=cfg["bidirectional"],
            dropout=cfg["dropout"] if cfg["num_layer"] > 1 else 0.0,
        )
        self.linear = nn.ModuleDict(
            {"0": nn.Linear(cfg["hidden_num"], 64), "2": nn.Linear(64, 1)}
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        mask = torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None]
        feat = (out * mask.unsqueeze(-1).float()).sum(dim=1) / lengths[:, None].clamp(min=1)
        feat = F.relu(self.linear["0"](feat))
        return self.linear["2"](feat)


def encode_batch(seqs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([min(len(s), MAX_LEN) for s in seqs], dtype=torch.long)
    rows = []
    for s in seqs:
        s = s[:MAX_LEN]
        row = [AA_TO_IDX.get(c, 0) for c in s]
        row.extend([0] * (MAX_LEN - len(row)))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long), lengths


def run_inference(model_path: str, test_csv: str, output_dir: str,
                  batch_size: int = 512, device_id: str = "0") -> str:
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() and device_id != "cpu" else "cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    model = LstmNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    df = pd.read_csv(test_csv)
    seqs = df["Sequence"].astype(str).tolist()
    preds = []
    with torch.no_grad():
        for start in tqdm(range(0, len(seqs), batch_size), desc="Inference"):
            batch_seqs = seqs[start: start + batch_size]
            x, lengths = encode_batch(batch_seqs)
            x, lengths = x.to(device), lengths.to(device)
            out = model(x, lengths).squeeze(-1).cpu().tolist()
            preds.extend(out)

    df = df.copy()
    df["predict_MIC"] = preds
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    out_path = os.path.join(output_dir, f"{model_name}_test_results.csv")
    df.to_csv(out_path, index=False)
    print(f"[Done] Saved {len(df)} rows -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    run_inference(args.model_path, args.test_csv, args.output_dir,
                  batch_size=args.batch_size, device_id=args.device)


if __name__ == "__main__":
    main()
