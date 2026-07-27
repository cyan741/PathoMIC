#!/usr/bin/env bash
# Run ONE Stage-3 fusion / h3-variant experiment with an explicit seed.
#
# Variants (all match the seed=42 Stage-3 runs on splits2):
#   f_bilinear    hier + bilinear fusion,  gnn_out_dim=64,  post
#   f_film        hier + film,             gnn_out_dim=640, post
#   f_gated       hier + gated,            gnn_out_dim=640, post
#   f_xattn       hier + cross_attn,       gnn_out_dim=640, post
#   h3_raw192     hier_raw + concat,       gnn_out_dim=64,  post  (192-d species vec)
#   h3_prefix_pep hier + prefix inject,    gnn_out_dim=640, prefix_pool=peptide
#   h3_prefix_all hier + prefix inject,    gnn_out_dim=640, prefix_pool=all
#
# Shared h3 hyperparams: GCN 2-layer, hidden=128, hier levels species,genus,family,
# ESM2-150M full FT, MSE, bs=32, lr=1e-5, 50 epochs, ES patience=10, es_min_epoch=20.
#
# Usage:
#   GPU=0 SEED=7  VARIANT=f_gated       bash gnn_150M_fusion_variant.sh
#   GPU=4 SEED=123 VARIANT=h3_prefix_pep SPLIT=splits2 bash gnn_150M_fusion_variant.sh

set -u
PLM="esm2-150M"
PY=${PY:-/home/luyq/miniconda3/envs/pytorch3.9/bin/python}
ROOT_DIR="/home/luyq/PLM_AMP_Regression/scripts"
cd "${ROOT_DIR}"

CKP_BASE="/NAS/luyq/PLM_AMP_Regression/gnn_runs/${PLM}"
LOG_BASE="${CKP_BASE}/logs"
mkdir -p "${CKP_BASE}" "${LOG_BASE}"
TAXO=/NAS/luyq/AMP_datasets/taxonomy_graph.pt

GPU=${GPU:?GPU env var is required}
SEED=${SEED:?SEED env var is required}
VARIANT=${VARIANT:?VARIANT required (f_bilinear|f_film|f_gated|f_xattn|h3_raw192|h3_prefix_pep|h3_prefix_all)}
SPLIT=${SPLIT:-splits2}
EPOCHS=${EPOCHS:-50}
BS=${BS:-32}
LR=${LR:-1e-5}

data="/NAS/luyq/AMP_datasets/${SPLIT}"
sub="${VARIANT}_${SPLIT}_sd${SEED}"
ckp="${CKP_BASE}/${sub}"
log="${LOG_BASE}/${sub}.log"
mkdir -p "${ckp}"

common_args=(
    --data_path "${data}"
    --plm "${PLM}"
    --finetune_plm True
    --seed "${SEED}"
    --species_mode gnn
    --taxo_graph_path "${TAXO}"
    --gnn_hidden 128 --gnn_layers 2
    --gnn_type gcn
    --gnn_hier_levels species,genus,family
    --batch_size "${BS}" --lr "${LR}" --epochs "${EPOCHS}"
    --use_scheduler --warmup_ratio 0.1 --min_lr_ratio 0.05
    --early_stopping --early_stop_patience 10 --es_min_epoch 20
    --save_dir "${ckp}"
    --metrics_name "${sub}.csv"
    --test_results_dir "${CKP_BASE}/fusion_variants_test_results"
    --device "${GPU}"
)

case "${VARIANT}" in
    f_bilinear)
        variant_args=(
            --gnn_out_dim 64 --gnn_fusion hier
            --fusion_strategy bilinear --species_inject post
        )
        ;;
    f_film)
        variant_args=(
            --gnn_out_dim 640 --gnn_fusion hier
            --fusion_strategy film --species_inject post
        )
        ;;
    f_gated)
        variant_args=(
            --gnn_out_dim 640 --gnn_fusion hier
            --fusion_strategy gated --species_inject post
        )
        ;;
    f_xattn)
        variant_args=(
            --gnn_out_dim 640 --gnn_fusion hier
            --fusion_strategy cross_attn --species_inject post
        )
        ;;
    h3_raw192)
        variant_args=(
            --gnn_out_dim 64 --gnn_fusion hier_raw
            --fusion_strategy concat --species_inject post
        )
        ;;
    h3_prefix_pep)
        variant_args=(
            --gnn_out_dim 640 --gnn_fusion hier
            --fusion_strategy concat --species_inject prefix --prefix_pool peptide
        )
        ;;
    h3_prefix_all)
        variant_args=(
            --gnn_out_dim 640 --gnn_fusion hier
            --fusion_strategy concat --species_inject prefix --prefix_pool all
        )
        ;;
    *)
        echo "Unknown VARIANT=${VARIANT}" >&2
        exit 2
        ;;
esac

echo "================ [${PLM}] ${sub} ================"
echo "  variant=${VARIANT} seed=${SEED} split=${SPLIT} gpu=${GPU}"
echo "  start: $(date)"
"${PY}" train.py \
    "${common_args[@]}" \
    "${variant_args[@]}" \
    > "${log}" 2>&1
rc=$?
echo "  done : $(date)  rc=${rc}"
exit ${rc}
