#!/bin/bash

# ===================================
# [사용법] bash eval.sh <dataset_name>
# 예시: bash eval.sh itw
# ===================================

# ========================
# 인자 확인
# ========================
if [ $# -ne 1 ]; then
  echo "❌ Usage: bash eval.sh <dataset_name>"
  echo "예: bash eval.sh itw"
  exit 1
fi

DATASET=$1

# ========================
# 설정 파일
# ========================
DATASET_YAML="/home/woongjae/wildspoof/SFM-ADD/configs/dataset_curriculum.yaml"
CONFIG_FILE="/home/woongjae/wildspoof/SFM-ADD/configs/conformertcm_curriculum.yaml"
MODEL_PATH="/home/woongjae/wildspoof/SFM-ADD/out/conformertcm_curriculum.pth"

# ========================
# YAML 파서(yq로 읽기)
# ========================
DATABASE_PATH=$(yq ".${DATASET}.database_path" ${DATASET_YAML})
PROTOCOL_PATH=$(yq ".${DATASET}.protocol_path" ${DATASET_YAML})
EVAL_OUTPUT=$(yq ".${DATASET}.eval_output" ${DATASET_YAML})

# 🔧 따옴표 제거 (sed 사용)
DATABASE_PATH=$(echo $DATABASE_PATH | sed 's/"//g')
PROTOCOL_PATH=$(echo $PROTOCOL_PATH | sed 's/"//g')
EVAL_OUTPUT=$(echo $EVAL_OUTPUT | sed 's/"//g')

# ========================
# 값 확인
# ========================
echo "=========================================="
echo "🚀 Dataset: ${DATASET}"
echo "📂 DATABASE_PATH: ${DATABASE_PATH}"
echo "📜 PROTOCOL_PATH: ${PROTOCOL_PATH}"
echo "💾 EVAL_OUTPUT: ${EVAL_OUTPUT}"
echo "=========================================="

# ========================
# 평가 실행
# ========================
CUDA_VISIBLE_DEVICES=MIG-57de94a5-be15-5b5a-b67e-e118352d8a59 python /home/woongjae/wildspoof/SFM-ADD/main.py \
  --eval \
  --database_path "${DATABASE_PATH}" \
  --protocol_path "${PROTOCOL_PATH}" \
  --config "${CONFIG_FILE}" \
  --model_path "${MODEL_PATH}" \
  --eval_output "${EVAL_OUTPUT}" \
  --batch_size 32
