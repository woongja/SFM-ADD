#!/bin/bash

# ===================================
# 모든 데이터셋 평가 자동 실행
# ===================================

DATASETS=("itw" "wildspoof" "deepen" "asv19_noise" "df21_noise")

for DATASET in "${DATASETS[@]}"; do
  echo "=========================================="
  echo "🚀 Running evaluation for: ${DATASET}"
  echo "=========================================="
  
  bash eval.sh "${DATASET}"
  
  # 오류 발생 시 중단
  if [ $? -ne 0 ]; then
    echo "❌ Error occurred while evaluating ${DATASET}. Stopping."
    exit 1
  fi
  
  echo "✅ Finished evaluation for ${DATASET}"
  echo ""
done

echo "🎉 All evaluations completed successfully!"
