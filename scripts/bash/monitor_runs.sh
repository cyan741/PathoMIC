#!/usr/bin/env bash
# Print a one-shot status summary across every gnn_runs/<plm>/<sub>/log.
# Filters out tqdm carriage-return spam and shows only the lines that matter.

LOG_GLOB=${1:-/NAS/luyq/PLM_AMP_Regression/gnn_runs/*/logs/*.log}

for log in $LOG_GLOB; do
    [ -f "$log" ] || continue
    name=$(basename "$log" .log)
    plm=$(basename "$(dirname "$(dirname "$log")")")
    # Strip carriage-return-only tqdm lines and pull final status.
    last_line=$(tr '\r' '\n' < "$log" | grep -E "(Epoch [0-9]+ (Train|Val) MSE|Test MSE|Early stopping|BUCKETED|new best|Saving)" | tail -3)
    train_lines=$(tr '\r' '\n' < "$log" | grep -cE "Epoch [0-9]+ Train MSE Loss")
    val_lines=$(tr '\r' '\n' < "$log" | grep -cE "Epoch [0-9]+ Val MSE Loss")
    has_test=$(tr '\r' '\n' < "$log" | grep -cE "^Test MSE Loss")
    es=$(tr '\r' '\n' < "$log" | grep -c "Early stopping")
    last_loss=$(tr '\r' '\n' < "$log" | grep -E "Epoch [0-9]+ Val MSE" | tail -1 | sed 's/.*Val MSE Loss: //')

    if [ "$has_test" -gt 0 ]; then
        status="DONE"
    elif [ "$es" -gt 0 ]; then
        status="ES (testing)"
    else
        status="run (ep=${train_lines}/${val_lines})"
    fi
    printf "%-12s | %-22s | %-22s | last_val=%s\n" "$plm" "$name" "$status" "${last_loss:-n/a}"
done
