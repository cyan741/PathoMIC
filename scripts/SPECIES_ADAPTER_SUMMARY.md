# Species Adapter 集成 — 行动总结

## 目标
将 `species_embeddings.pkl`（520 种菌，768-d PubMedBERT 向量）通过一个非线性 adapter 注入 MIC 回归模型，与 ESM2-8M 的序列输出 concat 后送入 MLP 回归头。最后一层不加激活函数。

## 改动文件

| 文件 | 变更 |
|---|---|
| `PLM_head.py` | 新增 `SpeciesAdapter`；`MLP` 支持 `gelu` |
| `plm_models.py` | `ESM2` 新增 `use_species / species_*` 参数；`forward` 支持 `species_emb` |
| `data_loader.py` | 新增 `load_species_embeddings`；`MIC_Dataset` 按 `Target_Species` 查 embedding（缺失→零向量）；`data_loader()` 新增 `species_emb_path` 参数 |
| `train.py` | 新增 `--use_species / --species_emb_path / --species_emb_dim / --species_out_dim / --species_bottleneck / --species_dropout`；`_unpack_batch` 辅助函数；默认 `data_path` 改为 `splits1` |

## 架构（ESM2-8M + SpeciesAdapter）

```
input_ids  [B, L=72]              species_emb  [B, 768]
    │                                   │
 ESM2-8M                         SpeciesAdapter
 (6-layer Transformer)           ┌─ Linear(768→128)
    │                            ├─ LayerNorm
 last_hidden [B, L, 320]         ├─ GELU
    │ mean-pool                  ├─ Dropout(0.1)
 seq_rep    [B, 320]             ├─ Linear(128→128)
    │                            └─ GELU
    │                                   │
    └──────── concat ───────────────────┘
              [B, 320 + 128 = 448]
                   │
              MLP head:  448 → 256 → 64 → 1
              （最后一层纯 Linear，无激活、无 BN，可输出负值）
                   │
              pred MIC  [B, 1]
```

## 运行命令（1 epoch，GPU 4）

```bash
cd /home/luyq/PLM_AMP_Regression/scripts
conda run -n pytorch3.9 python3 train.py \
    --epochs 1 --batch_size 32 --lr 1e-5 --device 4 \
    --use_species \
    --species_emb_path /home/luyq/AMP_datasets/species_embeddings.pkl \
    --data_path /home/luyq/AMP_datasets/splits1 \
    --save_dir /NAS/luyq/PLM_AMP_Regression/ckp_species_test \
    --metrics_name train_metrics_species_test.csv
```

## 验证结果
- 训练集 39274 行 / 25 个菌种；验证 4912、测试 4259。
- 520 个菌种嵌入全部匹配训练集物种，无缺失。
- Epoch-1 耗时约 **98 秒**（A40-40G），GPU 正常占用。
- **Train MSE 1.7819 | Val MSE 1.7846 | Test MSE 1.8243**。
- Checkpoint：`/NAS/luyq/PLM_AMP_Regression/ckp_species_test/esm2-8M_lr1e-05_bs32_ep1.pth`。

## 向后兼容
未加 `--use_species` 时，模型、dataset、train loop 自动退化为原始 (seq, mic) 两元组路径，不影响旧实验脚本。
