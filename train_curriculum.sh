#!/bin/bash

# ========================
# Curriculum Learning Training Script
# ========================

# 경로 및 설정
DATABASE_PATH="/home/woongjae/wildspoof/Datasets"
CONFIG_FILE="/home/woongjae/wildspoof/SFM-ADD/configs/conformertcm_curriculum.yaml"
PROTOCOL_PATH="/home/woongjae/wildspoof/protocols/protocol_wildspoof.txt"
MODEL_SAVE_PATH="out/conformertcm_curriculum_batch_22_early_3.pth"
COMMENT="conformertcm_curriculum_learning_batch_22_early_3"

# GPU 설정 (필요시 수정)
# CUDA_VISIBLE_DEVICES=0
# 또는 MIG 사용시:
# CUDA_VISIBLE_DEVICES=MIG-56c6e426-3d07-52cb-aa59-73892edacb69

# ========================
# Curriculum Learning 실행
# ========================
CUDA_VISIBLE_DEVICES=MIG-46b32d1b-f775-5b7d-a987-fb8ebc049494 python main.py \
  --database_path ${DATABASE_PATH} \
  --protocol_path ${PROTOCOL_PATH} \
  --config ${CONFIG_FILE} \
  --batch_size 22 \
  --num_epochs 100 \
  --max_lr 1e-4 \
  --weight_decay 1e-4 \
  --patience 3 \
  --seed 1234 \
  --model_save_path ${MODEL_SAVE_PATH} \
  --comment ${COMMENT} \
  --algo 3

# ========================
# 설정 설명
# ========================
# --database_path: 데이터셋 경로
# --protocol_path: 프로토콜 파일 경로 (train/dev/eval 구분)
# --config: Curriculum config 파일 (sfm_backend_curriculum.yaml)
# --batch_size: 24 (curriculum sampler에서 사용)
# --num_epochs: 총 epoch 수 (curriculum stages의 합보다 크게 설정)
# --max_lr: Learning rate
# --patience: Early stopping patience
# --algo: 0 (RawBoost 사용 안함 - curriculum에서는 online augmentation 사용)
# --comment: TensorBoard 로그 디렉토리 이름

# ========================
# Curriculum 설정 수정
# ========================
# configs/conformertcm_curriculum.yaml 에서 수정:
# - model: conformertcm (emb_size, heads, kernel_size, num_encoders)
# - stage1/2/3/4의 epochs 수 조절
# - 각 stage의 description 확인
#
# aug/augmentation_config_curriculum.yaml 에서 수정:
# - stage2/3/4의 SNR 범위 조절
# - 각 augmentation의 강도 파라미터 조절
