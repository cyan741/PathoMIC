#!/usr/bin/env bash
# Prefix-tuning / LoRA sweep on ESM2-150M (splits1, 5 seeds per variant).
#
# Variants:
#   P1       frozen + shallow prefix, pool=all
#   P1b      frozen + shallow prefix, pool=peptide
#   P2       LoRA r=8  + shallow prefix, pool=peptide
#   P3       LoRA r=16 + shallow prefix, pool=peptide
#   PDeep    frozen + deep prefix (per-layer KV), pool=peptide
#   BFrozen  frozen + concat640 post-fusion baseline
#   BLoRA8   LoRA r=8  + concat640 post-fusion baseline
#   BLoRA16  LoRA r=16 + concat640 post-fusion baseline
#
# Usage:
#   GPU=0 VARIANT=P1 SEED=42 bash gnn_150M_prefix_tune.sh

set -u
PLM="esm2-150M"
PY=${PY:-/home/luyq/miniconda3/envs/pytorch3.9/bin/python}
ROOT_DIR="/home/luyq/PLM_AMP_Regression/scripts"
cd "${ROOT_DIR}"

CKP_BASE="/NAS/luyq/PLM_AMP_Regression/gnn_runs/${PLM}"
LOG_BASE="${CKP_BASE}/logs"
mkdir -p "${CKP_BASE}" "${LOG_BASE}"
TAXO=/NAS/luyq/AMP_datasets/taxonomy_graph.pt

GPU=${GPU:?GPU required}
SEED=${SEED:?SEED required}
VARIANT=${VARIANT:?VARIANT required (P1|P1b|P2|P3|PDeep|BFrozen|BLoRA8|BLoRA16)}
SPLIT=${SPLIT:-splits1}

data="/NAS/luyq/AMP_datasets/${SPLIT}"
sub="${VARIANT}_${SPLIT}_sd${SEED}"
ckp="${CKP_BASE}/${sub}"
log="${LOG_BASE}/${sub}.log"
mkdir -p "${ckp}"

species_args=(
    --species_mode gnn
    --taxo_graph_path "${TAXO}"
    --gnn_hidden 128 --gnn_out_dim 640 --gnn_layers 2
    --gnn_type gcn --gnn_fusion hier
    --gnn_hier_levels species,genus,family
)

case "${VARIANT}" in
    P1)
        backbone_args=(--freeze_esm 1 --finetune_plm False)
        inject_args=(--species_inject prefix --prefix_depth shallow --prefix_pool all --fusion_strategy concat)
        LR=${LR:-5e-4}; GNN_MULT=${GNN_MULT:-1.0}; BS=${BS:-64}
        EPOCHS=${EPOCHS:-80}; ES_PAT=${ES_PAT:-15}; ES_MIN=${ES_MIN:-25}; WD=${WD:-1e-4}
        ;;
    P1b)
        backbone_args=(--freeze_esm 1 --finetune_plm False)
        inject_args=(--species_inject prefix --prefix_depth shallow --prefix_pool peptide --fusion_strategy concat)
        LR=${LR:-5e-4}; GNN_MULT=${GNN_MULT:-1.0}; BS=${BS:-64}
        EPOCHS=${EPOCHS:-80}; ES_PAT=${ES_PAT:-15}; ES_MIN=${ES_MIN:-25}; WD=${WD:-1e-4}
        ;;
    P2)
        backbone_args=(--use_lora 1 --lora_r 8 --lora_alpha 16 --lora_target query,value)
        inject_args=(--species_inject prefix --prefix_depth shallow --prefix_pool peptide --fusion_strategy concat)
        LR=${LR:-5e-4}; GNN_MULT=${GNN_MULT:-2.0}; BS=${BS:-32}
        EPOCHS=${EPOCHS:-60}; ES_PAT=${ES_PAT:-12}; ES_MIN=${ES_MIN:-20}; WD=${WD:-1e-4}
        ;;
    P3)
        backbone_args=(--use_lora 1 --lora_r 16 --lora_alpha 32 --lora_target query,value)
        inject_args=(--species_inject prefix --prefix_depth shallow --prefix_pool peptide --fusion_strategy concat)
        LR=${LR:-5e-4}; GNN_MULT=${GNN_MULT:-2.0}; BS=${BS:-32}
        EPOCHS=${EPOCHS:-60}; ES_PAT=${ES_PAT:-12}; ES_MIN=${ES_MIN:-20}; WD=${WD:-1e-4}
        ;;
    PDeep)
        backbone_args=(--freeze_esm 1 --finetune_plm False)
        inject_args=(--species_inject prefix --prefix_depth deep --prefix_pool peptide --prefix_kv_hidden 512 --fusion_strategy concat)
        LR=${LR:-5e-4}; GNN_MULT=${GNN_MULT:-1.0}; BS=${BS:-32}
        EPOCHS=${EPOCHS:-100}; ES_PAT=${ES_PAT:-20}; ES_MIN=${ES_MIN:-30}; WD=${WD:-1e-4}
        ;;
    BFrozen)
        backbone_args=(--freeze_esm 1 --finetune_plm False)
        inject_args=(--species_inject post --fusion_strategy concat)
        LR=${LR:-5e-4}; GNN_MULT=${GNN_MULT:-1.0}; BS=${BS:-64}
        EPOCHS=${EPOCHS:-80}; ES_PAT=${ES_PAT:-15}; ES_MIN=${ES_MIN:-25}; WD=${WD:-1e-4}
        ;;
    BLoRA8)
        backbone_args=(--use_lora 1 --lora_r 8 --lora_alpha 16 --lora_target query,value)
        inject_args=(--species_inject post --fusion_strategy concat)
        LR=${LR:-5e-4}; GNN_MULT=${GNN_MULT:-2.0}; BS=${BS:-32}
        EPOCHS=${EPOCHS:-60}; ES_PAT=${ES_PAT:-12}; ES_MIN=${ES_MIN:-20}; WD=${WD:-1e-4}
        ;;
    BLoRA16)
        backbone_args=(--use_lora 1 --lora_r 16 --lora_alpha 32 --lora_target query,value)
        inject_args=(--species_inject post --fusion_strategy concat)
        LR=${LR:-5e-4}; GNN_MULT=${GNN_MULT:-2.0}; BS=${BS:-32}
        EPOCHS=${EPOCHS:-60}; ES_PAT=${ES_PAT:-12}; ES_MIN=${ES_MIN:-20}; WD=${WD:-1e-4}
        ;;
    *)
        echo "Unknown VARIANT=${VARIANT}" >&2; exit 2 ;;
esac

echo "================ [${PLM}] ${sub} ================"
echo "  variant=${VARIANT} seed=${SEED} gpu=${GPU}"
echo "  lr=${LR} bs=${BS} gnn_lr_mult=${GNN_MULT} epochs=${EPOCHS} wd=${WD}"
echo "  start: $(date)"
"${PY}" train.py \
    --data_path "${data}" \
    --plm "${PLM}" \
    --seed "${SEED}" \
    "${species_args[@]}" \
    "${backbone_args[@]}" \
    "${inject_args[@]}" \
    --batch_size "${BS}" --lr "${LR}" --gnn_lr_mult "${GNN_MULT}" \
    --weight_decay "${WD}" \
    --epochs "${EPOCHS}" \
    --use_scheduler --warmup_ratio 0.05 --min_lr_ratio 0.05 \
    --early_stopping --early_stop_patience "${ES_PAT}" --es_min_epoch "${ES_MIN}" \
    --save_dir "${ckp}" \
    --metrics_name "${sub}.csv" \
    --test_results_dir "${CKP_BASE}/prefix_tune_test_results" \
    --device "${GPU}" \
    > "${log}" 2>&1
rc=$?
echo "  done : $(date)  rc=${rc}"
exit ${rc}
