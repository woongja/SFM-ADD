#!/bin/bash

# ===================================
# Balance SCL 모델 평가 스크립트 (파라미터 버전)
# [사용법] bash eval_balance_scl.sh <dataset_name> <gpu_id> <results_dir> <model_path> <config_file>
# 예시: bash eval_balance_scl.sh itw MIG-xxx /path/to/results /path/to/model.pth /path/to/config.yaml
# ===================================

# ========================
# 인자 확인
# ========================
if [ $# -ne 5 ]; then
  echo "❌ Usage: bash eval_balance_scl.sh <dataset_name> <gpu_id> <results_dir> <model_path> <config_file>"
  echo "예: bash eval_balance_scl.sh itw MIG-57de94a5-be15-5b5a-b67e-e118352d8a59 /home/woongjae/wildspoof/SFM-ADD/results/balance_scl /home/woongjae/wildspoof/SFM-ADD/out/conformertcm_balance_scl.pth /home/woongjae/wildspoof/SFM-ADD/configs/conformertcm_balance_scl.yaml"
  echo ""
  echo "Available datasets: itw, add2022, wildspoof, deepen, asv19_noise, df21_noise"
  exit 1
fi

DATASET=$1
GPU_ID=$2
RESULTS_DIR=$3
MODEL_PATH=$4
CONFIG_FILE=$5

# ========================
# 설정
# ========================
# 공통 데이터셋 정보
DATASET_YAML="/home/woongjae/wildspoof/SFM-ADD/configs/datasets_base.yaml"

# 결과 저장 경로 (자동 생성)
EVAL_OUTPUT="${RESULTS_DIR}/eval_${DATASET}.txt"

# ========================
# YAML 파서 (yq로 읽기)
# ========================
DATABASE_PATH=$(yq ".${DATASET}.database_path" ${DATASET_YAML})
PROTOCOL_PATH=$(yq ".${DATASET}.protocol_path" ${DATASET_YAML})

# 🔧 따옴표 제거
DATABASE_PATH=$(echo $DATABASE_PATH | sed 's/"//g')
PROTOCOL_PATH=$(echo $PROTOCOL_PATH | sed 's/"//g')

# ========================
# 값 확인
# ========================
if [ "$DATABASE_PATH" == "null" ] || [ "$PROTOCOL_PATH" == "null" ]; then
  echo "❌ Dataset '${DATASET}' not found in ${DATASET_YAML}"
  echo "Available datasets: itw, add2022, wildspoof, deepen, asv19_noise, df21_noise"
  exit 1
fi

# 결과 디렉토리 생성
mkdir -p ${RESULTS_DIR}

echo "=========================================="
echo "🚀 Balance SCL Model Evaluation"
echo "=========================================="
echo "📊 Dataset: ${DATASET}"
echo "📂 Database: ${DATABASE_PATH}"
echo "📜 Protocol: ${PROTOCOL_PATH}"
echo "🤖 Model: ${MODEL_PATH}"
echo "📝 Config: ${CONFIG_FILE}"
echo "💾 Output: ${EVAL_OUTPUT}"
echo "🎮 GPU: ${GPU_ID}"
echo "=========================================="

# ========================
# 평가 실행
# ========================
CUDA_VISIBLE_DEVICES=${GPU_ID} python /home/woongjae/wildspoof/SFM-ADD/main_scl.py \
  --eval \
  --database_path "${DATABASE_PATH}" \
  --protocol_path "${PROTOCOL_PATH}" \
  --config "${CONFIG_FILE}" \
  --model_path "${MODEL_PATH}" \
  --eval_output "${EVAL_OUTPUT}" \
  --batch_size 32

# ========================
# 결과 확인
# ========================
if [ $? -eq 0 ]; then
  echo "✅ Evaluation completed successfully!"
  echo "📊 Results saved to: ${EVAL_OUTPUT}"
else
  echo "❌ Evaluation failed!"
  exit 1
fi
