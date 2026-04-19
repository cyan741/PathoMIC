python train.py \
    --plm "esm2-8M" \
    --device 2 &

python train.py \
    --plm "esm2-35M" \
    --device 3 &

python train.py \
    --plm "esm2-150M" \
    --device 5 &

wait
echo "All training processes finished."
rm -r /NAS/luyq/PLM_AMP_Regression/wandb
echo "wandb cache cleared."