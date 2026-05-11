---
name: GNN-Only Optimal Search (v2)
overview: "Stage-wise ablation + sweep for the optimal TaxonomyGNN-only configuration. Addresses Q1 hier depth (incl. attention pooling), Q2 learnable init (incl. LoRA), Q3 fusion strategy (5 modes), and Q4 hyperparameters (lr/bs grid + GNN-specific lr + architecture + regularization + imbalance/OOD-aware losses). Each stage trains both splits1+splits2 and selects winner by mean test MSE; final stage uses 3 seeds + 3 control runs."
todos:
  - id: stage0_refactor
    content: "Stage 0: refactor gnn_module.py (configurable hier_levels + hier_attn fusion + LoRA init + residual/LN), add fusion_modules.py with 5 strategies, update plm_models.py dispatcher, train.py CLI, infer.py auto-detect"
    status: pending
  - id: stage0_loss_module
    content: "Stage 0b: add scripts/losses.py implementing Huber/SmoothL1, LDS-weighted (any base), Balanced MSE (BMC), Focal-R, GroupDRO; integrate into train.py via --loss_type and related args"
    status: pending
  - id: stage1_hier_depth
    content: "Stage 1: train h5/h7/hier_attn variants on splits1+splits2 (6 new runs); pick winner by mean test MSE"
    status: pending
  - id: stage2_learnable_init
    content: "Stage 2: with Stage 1 winner, train unfrozen (lr=1e-5, lr=5e-6) and LoRA-r16 init (6 runs); pick winner"
    status: pending
  - id: stage3_fusion
    content: "Stage 3: with Stage 1+2 winner, train 4 alt fusions (gated/FiLM/cross-attn/bilinear) + concat_out640 control on both splits (10 runs); pick winner"
    status: pending
  - id: stage4a_optimization
    content: "Stage 4a: lr x bs grid (3x3) on both splits (18 runs); pick best optimizer"
    status: pending
  - id: stage4b_gnn_lr
    content: "Stage 4b: GNN-vs-ESM differential lr_mult sweep {1,5,10,20} on both splits (4 runs incl. mult=1 reuse from 4a)"
    status: pending
  - id: stage4c_architecture
    content: "Stage 4c: 6 architecture combos (layers x hidden x out_dim) on both splits (12 runs); pick best architecture"
    status: pending
  - id: stage4d_arch_tricks
    content: "Stage 4d: residual+layernorm + dropout {0,0.1,0.2} on both splits (8 runs); pick best regularization"
    status: pending
  - id: stage4e_robust_loss
    content: "Stage 4e: robust loss family ¡ª Huber(delta=1.0) and SmoothL1 on both splits (4 runs); confirm robustness gain over MSE"
    status: pending
  - id: stage4f_imbalance_loss
    content: "Stage 4f: imbalance-aware losses ¡ª LDS-weighted Huber, Balanced MSE (BMC), Focal-R(gamma=2) on both splits (6 runs); pick best long-tail handler"
    status: pending
  - id: stage4g_ood_loss
    content: "Stage 4g: OOD-aware loss ¡ª GroupDRO with species-count-buckets as groups, on both splits (2 runs); compare against best of 4f"
    status: pending
  - id: stage4h_loss_combo
    content: "Stage 4h: best loss combo (e.g. Huber + LDS reweight, or BMC + GroupDRO) on both splits (4 runs); confirm whether stacking helps"
    status: pending
  - id: stage5_final
    content: "Stage 5: combine all winners; train 3 seeds + 3 controls (vanilla / adapter / both) on both splits (12 runs)"
    status: pending
  - id: stage6_inference_plots
    content: "Stage 6: batch inference + scatter/bar plots + aggregate summary.csv across all stage winners"
    status: pending
isProject: false
---

# GNN-Only Optimal Search Plan (v2)

## Goal & current best

- Setting: `species_mode=gnn` only (no adapter), GCN > GAT confirmed, hier > leaf confirmed.
- Current best: `v1.5` (GCN, 2 layers, hidden=128, out=64, hier-{species,genus,family}, MSE, lr=1e-5, bs=32) ¡ú splits1 test MSE 0.4035, splits2 0.1832, splits2 [5,20) bucket 0.2025.

## Selection criterion (per stage)

Average of splits1 and splits2 `val_best.pth` test MSE. Tie-breakers: splits2 [5,20) bucket ¡ú splits1 OOD bucket ¡ú total #params ¡ú convergence epoch.

---

## Stage 0 ¡ª Code refactoring

### 0.1 Generalize hier depth + attention pooling in `scripts/gnn_module.py`
- Replace hardcoded `HIER_LEVELS` with constructor arg `hier_levels: Tuple[str,...]`.
- Add `fusion='hier_attn'`: attention-pool over all 8 path nodes instead of concat.
- Add `use_lora_init` + `lora_rank`: low-rank residual on frozen PubMedBERT init (`init = frozen + A @ B`, A¡Ê[N,r], B¡Ê[r,768]).
- Add `use_residual` (skip connections in GCN), `use_layernorm` (LN before ReLU between GCN layers).

### 0.2 Fusion-strategy dispatcher in `scripts/plm_models.py` + new `scripts/fusion_modules.py`
- 5 modes: `concat` (current) | `gated` | `film` | `cross_attn` | `bilinear`
- For `gated`/`film`: GNN out_dim must equal ESM hidden dim (640 for 150M).

### 0.3 Loss dispatcher in new `scripts/losses.py`
- Implement: `mse`, `huber(delta)`, `smooth_l1(beta)`, `lds_weighted(base, kernel='gauss', sigma)`, `bmc(noise_var)`, `focal_r(gamma)`, `group_dro(groups, eta)`.
- Each loss is a `nn.Module` accepting `(y_pred, y_true, meta)` where `meta` carries species_id / bucket_id when needed.

### 0.4 New CLI args in `scripts/train.py`
- `--gnn_hier_levels` (csv list, default `species,genus,family`)
- `--fusion_strategy` (default `concat`)
- `--use_lora_init`, `--lora_rank` (default 16)
- `--gnn_residual`, `--gnn_layernorm`
- `--gnn_lr_mult` (default 1.0; multiplies the base lr only for GNN params)
- `--loss_type` (`mse`/`huber`/`smooth_l1`/`lds`/`bmc`/`focal_r`/`group_dro`)
- `--loss_huber_delta`, `--loss_lds_sigma`, `--loss_bmc_noise`, `--loss_focal_gamma`, `--loss_dro_eta`, `--loss_dro_group_by` (`species`/`bucket`)
- `--weight_decay`

### 0.5 Update `test_scripts/infer.py` auto-detection
- Detect from state_dict: hier_levels count (from `absent_emb.shape[0]`), fusion type (`fusion.gate.*` / `.gamma.*` / `.attn.*` / `.linear_e.*`), LoRA presence (`gnn.init_lora_A` etc).

---

## Stage 1 ¡ª Q1 hier depth + attention pooling (6 new runs)

Variants on top of v1.5:
- `h3` (= v1.5, reuse existing splits1+splits2 results)
- `h5` = + order, class
- `h7` = + phylum, kingdom (skip h8 = +domain since domain has only 2 values)
- `hier_attn` = attention-pool over all 8 path nodes

Total NEW runs: 3 depths ¡Á 2 splits = **6 runs**.

**Hypothesis**: deeper hier helps OOD (splits1) more than ID; attention pooling lets the model auto-pick the most informative ancestor levels.

---

## Stage 2 ¡ª Q2 learnable init (6 runs)

Take Stage 1 winner; vary init policy:
- `q2_unfrozen_lr1e-5_{splits1,splits2}`: 2 runs
- `q2_unfrozen_lr5e-6_{splits1,splits2}`: 2 runs (mitigate prior drift)
- `q2_lora_r16_{splits1,splits2}`: 2 runs (low-rank residual, retain prior)

**Necessity analysis**: splits1's 307 OOD species nodes never receive direct gradient; unfreezing only updates seen-species & internal nodes, which propagates via GCN. LoRA preserves PubMedBERT prior most safely.

---

## Stage 3 ¡ª Q3 fusion strategies (10 runs)

Build on Stage 1+2 winner. Note: `gated`/`film`/`cross_attn` need GNN out_dim = ESM hidden (640).

- `f_gated_{splits}`: 2 runs (sigmoid gate, dynamic balance ¡ª hypothesis: best for long-tail)
- `f_film_{splits}`: 2 runs (¦Ã,¦Â feature-level conditioning)
- `f_xattn_{splits}`: 2 runs (GNN attends to peptide tokens ¡ª info-richest)
- `f_bilinear_{splits}`: 2 runs (Hadamard interaction)
- `f_concat_out640_{splits}`: 2 runs (concat at out_dim=640 ¡ª control for "is the gain just from wider GNN?")

---

## Stage 4 ¡ª Q4 hyperparameter sweep (54 runs total)

All built on Stage 3 winner. Five sub-stages:

### 4a ¡ª Optimization (lr ¡Á bs grid, 18 runs)
- `lr ¡Ê {5e-6, 1e-5, 3e-5}` ¡Á `bs ¡Ê {16, 32, 64}` ¡Á 2 splits

### 4b ¡ª GNN/ESM differential lr (4 runs)
- `gnn_lr_mult ¡Ê {1, 5, 10, 20}` ¡Á 2 splits (mult=1 reused from 4a winner)
- Rationale: GNN has only ~300K params vs ESM's 150M; same lr is too slow.

### 4c ¡ª Architecture (12 runs)
- (L=2, h=128, out=current) ¡ª baseline
- (L=3, h=128, out=current) ¡ª deeper
- (L=4, h=128, out=current) ¡ª over-smooth check
- (L=2, h=256, out=2¡Ácurrent) ¡ª wider
- (L=3, h=256, out=2¡Ácurrent) ¡ª deeper+wider
- (L=2, h=512, out=4¡Ácurrent) ¡ª very wide
- ¡Á 2 splits

### 4d ¡ª Architecture tricks + regularization (8 runs)
- `+residual+layernorm` ¡Á 2 splits (2 runs)
- `gnn_dropout ¡Ê {0.0, 0.1, 0.2}` ¡Á 2 splits (6 runs; 0.1 may reuse if it's the winner)

### 4e ¡ª Robust loss (4 runs)
- `huber_delta=1.0` ¡Á 2 splits
- `smooth_l1_beta=1.0` ¡Á 2 splits
- Compare to MSE baseline (Stage 4d winner).

### 4f ¡ª Imbalance-aware loss (6 runs)
- `lds_weighted_huber(sigma=2.0)` ¡Á 2 splits
- `bmc(noise_var=1.0)` ¡Á 2 splits
- `focal_r(gamma=2.0)` ¡Á 2 splits
- Diagnostic: report bucketed test MSE; long-tail bucket [5,20) is the key metric.

### 4g ¡ª OOD-aware loss (2 runs)
- `group_dro(group_by=species_count_bucket, eta=0.01)` ¡Á 2 splits
- Compare directly with best of 4f.

### 4h ¡ª Loss combination (4 runs)
- Top-2 loss winners stacked (e.g. `lds_weighted_huber` + GroupDRO grouping)
- ¡Á 2 splits, plus 1 ablation per split disabling one component (4 runs total)

**4d¡ú4e¡ú4f¡ú4g¡ú4h** uses sequential winner.

---

## Stage 5 ¡ª Final candidate + controls (12 runs)

Combine all stage winners. Run with longer training:
- `epochs=80`, `patience=15`, `min_epoch=25`
- 3 seeds (`42`, `1337`, `2025`) ¡Á 2 splits = **6 runs**

Plus 3 controls (with the same hyperparams as the final winner):
- `species_mode=none` ¡Á 2 splits = 2 runs (vanilla ESM2-150M, isolate GNN contribution)
- `species_mode=adapter` ¡Á 2 splits = 2 runs (PubMedBERT adapter, current legacy baseline)
- `species_mode=both` ¡Á 2 splits = 2 runs (do GNN+adapter > GNN alone?)

Total: **12 runs**.

---

## Stage 6 ¡ª Aggregation, inference, plots

- Extend `test_scripts/run_all_gnn_infer.sh` to all stage winners + final 6-seed ensemble.
- Outputs: `test_results/tax_gnn/esm150m_v2/{stage}/{splits1,splits2,external}_test/{csv,scatter_figs,metrics_by_type}/`.
- `summary.csv` columns: `stage, variant, split, seed, val_best_mse, test_mse, bucket_lt5, bucket_5_20, bucket_20_100, bucket_ge100, spearman, kendall, n_params, train_hours`.
- Final plots:
  - Bar chart: stage progression (test MSE per split)
  - Bar chart: long-tail bucket comparison (final winner vs v1.5 vs vanilla ESM)
  - Scatter: best-of-stage-5 prediction vs ground truth on all 3 test sets

---

## Run accounting

| Stage | Runs |
|---|---|
| Stage 1 | 6 |
| Stage 2 | 6 |
| Stage 3 | 10 |
| Stage 4a | 18 |
| Stage 4b | 4 |
| Stage 4c | 12 |
| Stage 4d | 8 |
| Stage 4e | 4 |
| Stage 4f | 6 |
| Stage 4g | 2 |
| Stage 4h | 4 |
| Stage 5 | 12 |
| **Total** | **92** |

¡Ö 200¨C280 GPU-h on A100 (each run 2¨C3h). Inference + plots: ~30 min.

---

## Mermaid: stage flow

```mermaid
flowchart TD
    S0a[Stage 0a - Code refactor]
    S0b[Stage 0b - Loss module]
    S1[Stage 1 - hier h3/h5/h7/hier_attn]
    S2[Stage 2 - frozen vs unfrozen vs LoRA]
    S3[Stage 3 - 5 fusion strategies]
    S4a[Stage 4a - lr x bs]
    S4b[Stage 4b - GNN lr_mult]
    S4c[Stage 4c - layers/hidden/out]
    S4d[Stage 4d - residual/LN/dropout]
    S4e[Stage 4e - Huber/SmoothL1]
    S4f[Stage 4f - LDS/BMC/Focal-R]
    S4g[Stage 4g - GroupDRO]
    S4h[Stage 4h - loss combo]
    S5[Stage 5 - 3 seeds + 3 controls]
    S6[Stage 6 - inference + plots]

    S0a --> S0b --> S1 --> S2 --> S3 --> S4a --> S4b --> S4c --> S4d --> S4e --> S4f --> S4g --> S4h --> S5 --> S6