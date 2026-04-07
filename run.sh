#!/bin/bash

CMDS=(
# "bash run_TinyStress.sh --stage 0 --stop_stage 3"
# "bash run_TinyStress_gts.sh"
# "bash run_StressTest.sh --stop_stage 0"
# "bash run_StressPresso.sh --stop_stage 0"
# "bash run_Emphassess.sh --stop_stage 0"
"bash run_TinyStress_s.sh"
# "./run.sh --stage 3 --gpuid 0 --train_conf conf/baseline.json --eval_dataset TinyStress"
)

FAILED_CMDS=()

INTERVAL=300
MEM_THRESHOLD=20000

while true; do
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)

    READY_GPU=$(echo "$FREE_MEM" | awk -v threshold="$MEM_THRESHOLD" '$1>=threshold {print NR-1; exit}')

    if [ -n "$READY_GPU" ]; then
        echo "[$(date)] GPU $READY_GPU 有足夠空間，開始執行程式..."

        for CMD in "${CMDS[@]}"; do
            echo "Running: $CMD"
            eval "$CMD --gpuid $READY_GPU" | tee log.txt || { echo "失敗: $CMD"; FAILED_CMDS+=("$CMD"); }
        done
        echo "====== 執行完成 ======"
        echo "失敗的指令："
        printf '%s\n' "${FAILED_CMDS[@]}"
        break
    else
        echo "[$(date)] 沒有 GPU 空間足夠，等待 $INTERVAL 秒..."
        sleep $INTERVAL
    fi
done