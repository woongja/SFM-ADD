#!/bin/bash

# ================================================================================
# Training Script for Balanced Training with Supervised Contrastive Learning
# ================================================================================
# Usage:
#   ./train_balance_scl.sh [GPU_ID]
#
# Examples:
#   ./train_balance_scl.sh MIG-8cdeef83-092c-5a8d-a748-452f299e1df0
#   ./train_balance_scl.sh
# ================================================================================

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Common settings
DATABASE_PATH="/home/woongjae/wildspoof/Datasets"
DEFAULT_GPU="MIG-8cdeef83-092c-5a8d-a748-452f299e1df0"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Get GPU ID from argument or use default
GPU_ID=MIG-57de94a5-be15-5b5a-b67e-e118352d8a59
# Training configuration
PROTOCOL_PATH="/home/woongjae/wildspoof/protocols/protocol_wildspoof_n.txt"
CONFIG_FILE="configs/conformertcm_balance_scl.yaml"
MODEL_SAVE_PATH="out/conformertcm_balance_scl_weight_loss.pth"
COMMENT="conformertcm_balance_scl_augaware_contrastive"

# Print settings
echo ""
print_info "==================================================================="
print_info "  Balanced Training with Supervised Contrastive Learning (SCL)   "
print_info "==================================================================="
echo ""
print_info "GPU Device: ${GPU_ID}"
print_info "Database Path: ${DATABASE_PATH}"
print_info "Protocol: ${PROTOCOL_PATH}"
print_info "Config: ${CONFIG_FILE}"
print_info "Model Save Path: ${MODEL_SAVE_PATH}"
print_info "Comment: ${COMMENT}"
echo ""
print_info "Batch composition:"
print_info "  - 10 augmentation types × 2 (bonafide + spoof) = 20 samples"
print_info "  - 2 clean samples (1 bonafide + 1 spoof) = 2 samples"
print_info "  - Total: 22 samples per batch"
echo ""
print_info "Augmentation-aware contrastive learning:"
# print_info "  - Same label + Same augmentation → weight = 0.3 (easy pairs)"
# print_info "  - Same label + Different augmentation → weight = 1.0 (hard pairs)"
echo ""
print_info "Starting training..."
echo ""

# Run training
CUDA_VISIBLE_DEVICES=${GPU_ID} python main_scl.py \
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

# Check exit status
if [ $? -eq 0 ]; then
    print_success "Training completed successfully!"
else
    print_error "Training failed!"
    exit 1
fi
