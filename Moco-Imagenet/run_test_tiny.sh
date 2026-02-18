#!/bin/bash
# Full training on imagenet100_tiny (5 classes, 250 train images).
# Runs 3 augmentation configs with real hyperparameters, logs output, generates CSV.
set -e

DATA="./imagenet100_tiny"
BASE_DIR="./checkpoints/tiny_test"
EPOCHS_PRETRAIN=500
EPOCHS_LINCLS=200
BATCH_SIZE=64
MOCO_K=128
WORKERS=2
ARCH="resnet50"
LOG="tiny_test.log"
CSV="tiny_results.csv"

# Write CSV header
echo "experiment,arch,epochs_pretrain,epochs_lincls,batch_size,moco_k,moco_m,moco_t,moco_dim,lr_pretrain,lr_lincls,use_color,use_rotation,object_acc1,object_acc5,rotation_acc1,rotation_acc4" > ${CSV}

echo "==================================================" | tee ${LOG}
echo "  Full Training on imagenet100_tiny — 3 experiments" | tee -a ${LOG}
echo "==================================================" | tee -a ${LOG}
date | tee -a ${LOG}

# --- Experiment configs ---
# name        use_color  use_rotation
EXPERIMENTS=(
    "exp_color           --use-color  "
    "exp_rotation        --no-color   --use-rotation"
    "exp_color_rotation  --use-color  --use-rotation"
)

for exp_line in "${EXPERIMENTS[@]}"; do
    # Parse experiment name and flags
    read -r EXP_NAME EXP_FLAGS <<< "${exp_line}"
    SAVE_DIR="${BASE_DIR}/${EXP_NAME}"

    echo "" | tee -a ${LOG}
    echo "==================================================" | tee -a ${LOG}
    echo "  Experiment: ${EXP_NAME}" | tee -a ${LOG}
    echo "  Flags: ${EXP_FLAGS}" | tee -a ${LOG}
    echo "==================================================" | tee -a ${LOG}

    # Step 1: Pre-train MoCo
    echo "" | tee -a ${LOG}
    echo ">>> [${EXP_NAME}] Pre-Training (${EPOCHS_PRETRAIN} epochs)" | tee -a ${LOG}
    python main_moco.py \
        --data ${DATA} \
        --arch ${ARCH} \
        --epochs ${EPOCHS_PRETRAIN} \
        --batch-size ${BATCH_SIZE} \
        --moco-k ${MOCO_K} \
        --workers ${WORKERS} \
        --save-dir ${SAVE_DIR} \
        --save-freq ${EPOCHS_PRETRAIN} \
        --print-freq 1 \
        ${EXP_FLAGS} \
        2>&1 | tee -a ${LOG}

    CHECKPOINT="${SAVE_DIR}/checkpoint_$(printf '%04d' ${EPOCHS_PRETRAIN}).pth.tar"

    # Step 2: Linear eval — object classification
    echo "" | tee -a ${LOG}
    echo ">>> [${EXP_NAME}] Linear Eval — Object Classification (${EPOCHS_LINCLS} epochs)" | tee -a ${LOG}
    OBJ_OUTPUT=$(python main_lincls.py \
        --data ${DATA} \
        --arch ${ARCH} \
        --pretrained ${CHECKPOINT} \
        --epochs ${EPOCHS_LINCLS} \
        --batch-size ${BATCH_SIZE} \
        --workers ${WORKERS} \
        --print-freq 1 \
        2>&1)
    echo "${OBJ_OUTPUT}" | tee -a ${LOG}

    # Parse object classification results
    OBJ_ACC1=$(echo "${OBJ_OUTPUT}" | grep "Best Val Acc@1" | grep -oP '[\d.]+(?=%)')
    OBJ_ACC5=$(echo "${OBJ_OUTPUT}" | grep "Val Acc@5" | tail -1 | grep -oP 'Val Acc@5: \K[\d.]+')

    # Step 3: Linear eval — rotation classification
    echo "" | tee -a ${LOG}
    echo ">>> [${EXP_NAME}] Linear Eval — Rotation Classification (${EPOCHS_LINCLS} epochs)" | tee -a ${LOG}
    ROT_OUTPUT=$(python main_lincls.py \
        --data ${DATA} \
        --arch ${ARCH} \
        --pretrained ${CHECKPOINT} \
        --eval-rotation \
        --epochs ${EPOCHS_LINCLS} \
        --batch-size ${BATCH_SIZE} \
        --workers ${WORKERS} \
        --print-freq 1 \
        2>&1)
    echo "${ROT_OUTPUT}" | tee -a ${LOG}

    # Parse rotation classification results
    ROT_ACC1=$(echo "${ROT_OUTPUT}" | grep "Best Val Acc@1" | grep -oP '[\d.]+(?=%)')
    ROT_ACC4=$(echo "${ROT_OUTPUT}" | grep "Val Acc@5" | tail -1 | grep -oP 'Val Acc@5: \K[\d.]+')

    # Determine use_color and use_rotation flags for CSV
    USE_COLOR="True"
    USE_ROTATION="False"
    if echo "${EXP_FLAGS}" | grep -q "\-\-no-color"; then USE_COLOR="False"; fi
    if echo "${EXP_FLAGS}" | grep -q "\-\-use-rotation"; then USE_ROTATION="True"; fi

    # Append to CSV
    echo "${EXP_NAME},${ARCH},${EPOCHS_PRETRAIN},${EPOCHS_LINCLS},${BATCH_SIZE},${MOCO_K},0.999,0.2,128,0.03,30.0,${USE_COLOR},${USE_ROTATION},${OBJ_ACC1},${OBJ_ACC5},${ROT_ACC1},${ROT_ACC4}" >> ${CSV}

    echo "" | tee -a ${LOG}
    echo ">>> [${EXP_NAME}] Done — Object Acc@1: ${OBJ_ACC1}%  Rotation Acc@1: ${ROT_ACC1}%" | tee -a ${LOG}
done

echo "" | tee -a ${LOG}
echo "==================================================" | tee -a ${LOG}
echo "  All experiments complete!" | tee -a ${LOG}
echo "  Log:     ${LOG}" | tee -a ${LOG}
echo "  Results: ${CSV}" | tee -a ${LOG}
echo "==================================================" | tee -a ${LOG}
date | tee -a ${LOG}

echo ""
echo "--- CSV Contents ---"
column -t -s',' ${CSV}
