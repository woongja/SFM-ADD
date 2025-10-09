#!/bin/bash

# ========================
# 경로 및 설정
# ========================
DATABASE_PATH="/home/woongjae/wildspoof/Datasets"   # 실제 데이터셋 경로
CONFIG_FILE="/home/woongjae/wildspoof/SFM-ADD/configs/sfm_backend.yaml" # 설정 파일
protocol_path="/home/woongjae/wildspoof/protocol/protocol.txt"
MODEL_PATH="/home/woongjae/wildspoof/SFM-ADD/out/1.pth"  # 학습된 모델 경로
EVAL_OUTPUT="eval_scores.txt"  # 평가 결과 저장 경로

# ========================
# 평가 실행
# ========================
CUDA_VISIBLE_DEVICES=2 python main.py \
  --eval \
  --database_path ${DATABASE_PATH} \
  --protocol_path ${protocol_path} \
  --config ${CONFIG_FILE} \
  --model_path ${MODEL_PATH} \
  --eval_output ${EVAL_OUTPUT} \
  --batch_size 128
