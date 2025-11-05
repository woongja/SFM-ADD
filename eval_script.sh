#!/bin/bash

# ===================================
# Evaluation Script
# 여기에 파라미터만 수정해서 사용하세요
# ===================================

# ===================================
# 1. Curriculum Learning Model
# ===================================
GPU_ID="MIG-8cdeef83-092c-5a8d-a748-452f299e1df0"
RESULTS_DIR="/home/woongjae/wildspoof/SFM-ADD/results/curriculum"
MODEL_PATH="/home/woongjae/wildspoof/SFM-ADD/out/conformertcm_curriculum_batch_22_early_3.pth"
CONFIG_FILE="/home/woongjae/wildspoof/SFM-ADD/configs/conformertcm_curriculum.yaml"

bash eval_all_curriculum.sh "${GPU_ID}" "${RESULTS_DIR}" "${MODEL_PATH}" "${CONFIG_FILE}"

# ===================================
# 2. Balance Model (SCL 없음)
# ===================================
# GPU_ID="MIG-6e4275af-2db0-51f1-a601-7ad8a1002745"
# RESULTS_DIR="/home/woongjae/wildspoof/SFM-ADD/results/balance"
# MODEL_PATH="/home/woongjae/wildspoof/SFM-ADD/out/conformertcm_balance_22.pth"
# CONFIG_FILE="/home/woongjae/wildspoof/SFM-ADD/configs/conformertcm_balance.yaml"

# bash eval_all_balance.sh "${GPU_ID}" "${RESULTS_DIR}" "${MODEL_PATH}" "${CONFIG_FILE}"

# ===================================
# 3. Balance SCL Model (SCL 있음)
# ===================================
# GPU_ID="MIG-57de94a5-be15-5b5a-b67e-e118352d8a59"
# RESULTS_DIR="/home/woongjae/wildspoof/SFM-ADD/results/balance_scl"
# MODEL_PATH="/home/woongjae/wildspoof/SFM-ADD/out/conformertcm_balance_scl.pth"
# CONFIG_FILE="/home/woongjae/wildspoof/SFM-ADD/configs/conformertcm_balance_scl.yaml"

# bash eval_all_balance_scl.sh "${GPU_ID}" "${RESULTS_DIR}" "${MODEL_PATH}" "${CONFIG_FILE}"