#!/bin/bash

# ===================================
# Curriculum Learning 모델 - 모든 데이터셋 평가 자동 실행 (파라미터 버전)
# [사용법] bash eval_all_curriculum.sh <gpu_id> <results_dir> <model_path> <config_file>
# ===================================

# ========================
# 인자 확인
# ========================
if [ $# -ne 4 ]; then
  echo "❌ Usage: bash eval_all_curriculum.sh <gpu_id> <results_dir> <model_path> <config_file>"
  echo "예: bash eval_all_curriculum.sh MIG-8cdeef83-092c-5a8d-a748-452f299e1df0 /home/woongjae/wildspoof/SFM-ADD/results/curriculum /home/woongjae/wildspoof/SFM-ADD/out/conformertcm_curriculum.pth /home/woongjae/wildspoof/SFM-ADD/configs/conformertcm_curriculum.yaml"
  exit 1
fi

GPU_ID=$1
RESULTS_DIR=$2
MODEL_PATH=$3
CONFIG_FILE=$4

echo "=========================================="
echo "🚀 Curriculum Learning Model - Evaluating All Datasets"
echo "=========================================="
echo "🎮 GPU: ${GPU_ID}"
echo "📁 Results: ${RESULTS_DIR}"
echo "🤖 Model: ${MODEL_PATH}"
echo "📝 Config: ${CONFIG_FILE}"
echo "=========================================="
echo ""

DATASETS=("itw" "wildspoof" "deepen" "asv19_noise" "df21_noise")

for DATASET in "${DATASETS[@]}"; do
  echo "=========================================="
  echo "🔍 Evaluating: ${DATASET}"
  echo "=========================================="

  bash eval_curriculum.sh "${DATASET}" "${GPU_ID}" "${RESULTS_DIR}" "${MODEL_PATH}" "${CONFIG_FILE}"

  # 오류 발생 시 중단
  if [ $? -ne 0 ]; then
    echo "❌ Error occurred while evaluating ${DATASET}. Stopping."
    exit 1
  fi

  echo "✅ Finished evaluation for ${DATASET}"
  echo ""
done

echo "=========================================="
echo "🎉 All evaluations completed successfully!"
echo "=========================================="
echo ""
echo "📊 Results saved in: ${RESULTS_DIR}/"
