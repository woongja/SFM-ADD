#!/bin/bash

# ========================
# 경로 및 설정
# ========================
DATABASE_PATH="/home/woongjae/wildspoof/Datasets"   # 실제 데이터셋 경로
CONFIG_FILE="/home/woongjae/wildspoof/SFM-ADD/configs/sfm_backend.yaml" # 설정 파일
COMMENT="sfm_add_train"
protocol_path="/home/woongjae/wildspoof/protocol/protocol.txt"
MODEL_SAVE_PATH="out/best_model.pth"  # 모델 저장 경로

# ========================
# 훈련 실행
# ========================
CUDA_VISIBLE_DEVICES=MIG-8cdeef83-092c-5a8d-a748-452f299e1df0 python main.py \
  --database_path ${DATABASE_PATH} \
  --protocol_path ${protocol_path} \
  --config ${CONFIG_FILE} \
  --batch_size 64 \
  --num_epochs 100 \
  --min_lr 1e-7 \
  --max_lr 1e-4 \
  --weight_decay 1e-4 \
  --patience 5 \
  --seed 1234 \
  --model_save_path ${MODEL_SAVE_PATH} \
  --comment ${COMMENT}
