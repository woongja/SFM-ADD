#!/bin/bash

# ========================
# Balanced Training Script - All Augmentation Types in Each Batch
# ========================

# 경로 및 설정
DATABASE_PATH="/home/woongjae/wildspoof/Datasets"
CONFIG_FILE="/home/woongjae/wildspoof/SFM-ADD/configs/conformertcm_balance.yaml"
PROTOCOL_PATH="/home/woongjae/wildspoof/protocols/protocol_wildspoof_n.txt"
MODEL_SAVE_PATH="out/conformertcm_balance_22.pth"
COMMENT="conformertcm_balance_all_aug_types"

# GPU 설정 (필요시 수정)
# CUDA_VISIBLE_DEVICES=0
# 또는 MIG 사용시:
# CUDA_VISIBLE_DEVICES=MIG-56c6e426-3d07-52cb-aa59-73892edacb69

# ========================
# Balanced Training 실행
# ========================
CUDA_VISIBLE_DEVICES=MIG-46b32d1b-f775-5b7d-a987-fb8ebc049494 python main.py \
  --database_path ${DATABASE_PATH} \
  --protocol_path ${PROTOCOL_PATH} \
  --config ${CONFIG_FILE} \
  --batch_size 22 \
  --num_epochs 100 \
  --max_lr 1e-4 \
  --weight_decay 1e-4 \
  --patience 5 \
  --seed 1234 \
  --model_save_path ${MODEL_SAVE_PATH} \
  --comment ${COMMENT} \
  --algo 3

# ========================
# 설정 설명
# ========================
# --database_path: 데이터셋 경로
# --protocol_path: 프로토콜 파일 경로 (train/dev/eval 구분)
# --config: Balanced config 파일 (conformertcm_balance.yaml)
# --batch_size: 22 (10 aug types × 2 + 2 clean)
# --num_epochs: 총 epoch 수
# --max_lr: Learning rate
# --patience: Early stopping patience
# --algo: 0 (RawBoost 사용 안함 - online augmentation 사용)
# --comment: TensorBoard 로그 디렉토리 이름
#
# ========================
# Balanced Training 전략
# ========================
# 각 미니배치 구성:
#   - 10개 증강 타입 × 2 (bonafide + spoof) = 20 samples
#   - 2개 clean (bonafide + spoof) = 2 samples
#   총 22 samples
#
# 장점:
#   1. 모든 증강 타입이 매 배치에 포함됨
#   2. Bonafide/Spoof 비율 균형 (1:1)
#   3. 다양한 증강을 동시에 학습
