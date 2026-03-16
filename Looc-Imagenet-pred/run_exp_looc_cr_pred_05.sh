#!/bin/bash
# Experiment: LooC color+rotation + augmentation prediction classifier (lambda=0.5)
set -e

DATA="./imagenet100"
SAVE_DIR="./checkpoints/exp_looc_cr_pred_05"
LOG="exp_looc_cr_pred_05.log"
CSV="exp_looc_cr_pred_05_results.csv"
ARCH="resnet50"
EPOCHS_PRETRAIN=500
EPOCHS_LINCLS=200

echo "Starting exp_looc_cr_pred_05 (LooC + pred, lambda=0.5)" | tee ${LOG}
echo "Log: ${LOG}" | tee -a ${LOG}
date | tee -a ${LOG}

# Write CSV header
echo "experiment,arch,epochs_pretrain,epochs_lincls,batch_size,moco_k,moco_m,moco_t,moco_dim,lr_pretrain,lr_lincls,aug_types,lambda_pred,object_acc1,object_acc5,rotation_acc1,rotation_acc4" > ${CSV}

# Step 1: LooC Pre-Training (500 epochs)
echo "" | tee -a ${LOG}
echo ">>> Step 1: LooC Pre-Training (${EPOCHS_PRETRAIN} epochs, lambda_pred=0.5)" | tee -a ${LOG}
python main_looc.py \
    --data ${DATA} \
    --arch ${ARCH} \
    --epochs ${EPOCHS_PRETRAIN} \
    --batch-size 256 \
    --lr 0.03 \
    --schedule 300 400 \
    --moco-k 16384 \
    --moco-t 0.2 \
    --aug-types rotation color \
    --lambda-pred 0.5 \
    --save-dir ${SAVE_DIR} \
    --save-freq 50 \
    --workers 12 \
    2>&1 | tee -a ${LOG}

CHECKPOINT="${SAVE_DIR}/checkpoint_0500.pth.tar"

# Step 2: Linear eval — object classification (100 classes)
echo "" | tee -a ${LOG}
echo ">>> Step 2: Linear Eval — Object Classification" | tee -a ${LOG}
OBJ_OUTPUT=$(python main_lincls.py \
    --data ${DATA} \
    --arch ${ARCH} \
    --pretrained ${CHECKPOINT} \
    --looc-backbone \
    --epochs ${EPOCHS_LINCLS} \
    --batch-size 256 \
    --workers 12 \
    2>&1)
echo "${OBJ_OUTPUT}" | tee -a ${LOG}

# Parse object classification results
OBJ_ACC1=$(echo "${OBJ_OUTPUT}" | grep "Best Val Acc@1" | grep -oP '[\d.]+(?=%)')
OBJ_ACC5=$(echo "${OBJ_OUTPUT}" | grep "Val Acc@5" | tail -1 | grep -oP 'Val Acc@5: \K[\d.]+')

# Step 3: Linear eval — rotation classification (4 classes)
echo "" | tee -a ${LOG}
echo ">>> Step 3: Linear Eval — Rotation Classification" | tee -a ${LOG}
ROT_OUTPUT=$(python main_lincls.py \
    --data ${DATA} \
    --arch ${ARCH} \
    --pretrained ${CHECKPOINT} \
    --looc-backbone \
    --eval-rotation \
    --epochs ${EPOCHS_LINCLS} \
    --batch-size 256 \
    --workers 12 \
    2>&1)
echo "${ROT_OUTPUT}" | tee -a ${LOG}

# Parse rotation classification results
ROT_ACC1=$(echo "${ROT_OUTPUT}" | grep "Best Val Acc@1" | grep -oP '[\d.]+(?=%)')
ROT_ACC4=$(echo "${ROT_OUTPUT}" | grep "Val Acc@5" | tail -1 | grep -oP 'Val Acc@5: \K[\d.]+')

# Append to CSV
echo "exp_looc_cr_pred_05,${ARCH},${EPOCHS_PRETRAIN},${EPOCHS_LINCLS},256,16384,0.999,0.2,128,0.03,30.0,rotation+color,0.5,${OBJ_ACC1},${OBJ_ACC5},${ROT_ACC1},${ROT_ACC4}" >> ${CSV}

echo "" | tee -a ${LOG}
echo "exp_looc_cr_pred_05 COMPLETE" | tee -a ${LOG}
echo "Object Acc@1: ${OBJ_ACC1}%  Rotation Acc@1: ${ROT_ACC1}%" | tee -a ${LOG}
date | tee -a ${LOG}

echo ""
echo "--- CSV Contents ---"
column -t -s',' ${CSV}
