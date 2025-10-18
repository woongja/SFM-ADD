#!/bin/bash

# ========================
# 경로 및 설정
# ========================
DATABASE_PATH="/home/woongjae/wildspoof/Datasets"
CONFIG_FILE="/home/woongjae/wildspoof/SFM-ADD/configs/conformertcm_baseline.yaml"
PROTOCOL_PATH="/home/woongjae/wildspoof/protocols/protocol_wildspoof.txt"
MODEL_SAVE_PATH="out/conformertcm.pth"
# COMMENT="conformertcm_balanced_training"
COMMENT="conformertcmd_training"
# --batch_size 24
# ========================
# Balanced Training 실행
# ========================
CUDA_VISIBLE_DEVICES=MIG-56c6e426-3d07-52cb-aa59-73892edacb69 python balance_training.py \
  --database_path ${DATABASE_PATH} \
  --protocol_path ${PROTOCOL_PATH} \
  --config ${CONFIG_FILE} \
  --batch_size 32 \
  --num_epochs 100 \
  --max_lr 1e-6 \
  --weight_decay 1e-4 \
  --patience 10 \
  --seed 1234 \
  --model_save_path ${MODEL_SAVE_PATH} \
  --comment ${COMMENT} \
  --algo 3