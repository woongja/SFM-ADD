#!/bin/bash

# ========================
# GPU 설정
# ========================
export CUDA_VISIBLE_DEVICES=0   # 사용할 GPU 번호 (필요시 수정)

# ========================
# 경로 및 설정
# ========================
DATABASE_PATH="/AISRC2/Dataset"   # 실제 데이터셋 경로
CONFIG_FILE="configs/config.yaml" # 설정 파일
COMMENT="sfm_add_train"

# ========================
# 훈련 실행
# ========================
python main.py \
  --database_path ${DATABASE_PATH} \
  --config ${CONFIG_FILE} \
  --batch_size 8 \
  --num_epochs 100 \
  --min_lr 1e-7 \
  --max_lr 1e-4 \
  --weight_decay 1e-4 \
  --seed 1234 \
  --comment ${COMMENT}
