#!/bin/bash

# ================================================================================
# Unified Training Script for SFM-ADD Project
# ================================================================================
# Usage:
#   ./run_training.sh [MODE] [GPU_ID]
#
# Examples:
#   ./run_training.sh curriculum              # Curriculum learning (balanced)
#   ./run_training.sh curriculum_random       # Curriculum learning (random sampling)
#   ./run_training.sh balance                 # Balanced training (all aug in batch)
#   ./run_training.sh baseline                # Baseline training
#   ./run_training.sh                         # Interactive menu
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

# Function to show menu
show_menu() {
    echo ""
    echo "================================================================================"
    echo "                    SFM-ADD Training Mode Selection"
    echo "================================================================================"
    echo ""
    echo "  1) Curriculum Learning (Balanced Sampling)"
    echo "     - Per-stage early stopping"
    echo "     - Bonafide:Spoof = 1:1 in each batch"
    echo ""
    echo "  2) Curriculum Learning (Random Sampling)"
    echo "     - Per-stage early stopping"
    echo "     - Natural data distribution"
    echo ""
    echo "  3) Balanced Training (All Augmentations in Each Batch)"
    echo "     - 10 aug types × 2 + 2 clean = 22 samples per batch"
    echo "     - SNR range: 5-35 dB (light to strong)"
    echo ""
    echo "  4) Baseline Training"
    echo "     - Standard training without special sampling"
    echo ""
    echo "  5) Test Augmentations"
    echo "     - Test all augmentation types and generate samples"
    echo ""
    echo "  6) Exit"
    echo ""
    echo "================================================================================"
}

# Function to run curriculum training
run_curriculum() {
    print_info "Starting Curriculum Learning (Balanced Sampling)..."

    PROTOCOL_PATH="/home/woongjae/wildspoof/protocols/protocol_wildspoof.txt"
    CONFIG_FILE="configs/conformertcm_curriculum.yaml"
    MODEL_SAVE_PATH="out/conformertcm_curriculum.pth"
    COMMENT="conformertcm_curriculum_learning"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        --database_path ${DATABASE_PATH} \
        --protocol_path ${PROTOCOL_PATH} \
        --config ${CONFIG_FILE} \
        --batch_size 24 \
        --num_epochs 100 \
        --max_lr 1e-4 \
        --weight_decay 1e-4 \
        --patience 10 \
        --seed 1234 \
        --model_save_path ${MODEL_SAVE_PATH} \
        --comment ${COMMENT} \
        --algo 0
}

# Function to run curriculum random training
run_curriculum_random() {
    print_info "Starting Curriculum Learning (Random Sampling)..."

    PROTOCOL_PATH="/home/woongjae/wildspoof/protocols/protocol_wildspoof.txt"
    CONFIG_FILE="configs/conformertcm_curriculum_random.yaml"
    MODEL_SAVE_PATH="out/conformertcm_curriculum_random.pth"
    COMMENT="conformertcm_curriculum_random_sampling"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        --database_path ${DATABASE_PATH} \
        --protocol_path ${PROTOCOL_PATH} \
        --config ${CONFIG_FILE} \
        --batch_size 24 \
        --num_epochs 100 \
        --max_lr 1e-4 \
        --weight_decay 1e-4 \
        --patience 10 \
        --seed 1234 \
        --model_save_path ${MODEL_SAVE_PATH} \
        --comment ${COMMENT} \
        --algo 0
}

# Function to run balanced training
run_balance() {
    print_info "Starting Balanced Training (All Augmentations in Batch)..."

    PROTOCOL_PATH="/home/woongjae/wildspoof/protocols/protocol_wildspoof_n.txt"
    CONFIG_FILE="configs/conformertcm_balance.yaml"
    MODEL_SAVE_PATH="out/conformertcm_balance_22.pth"
    COMMENT="conformertcm_balance_all_aug_types"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
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
}

# Function to run baseline training
run_baseline() {
    print_info "Starting Baseline Training..."

    PROTOCOL_PATH="/home/woongjae/wildspoof/protocols/protocol_wildspoof.txt"
    CONFIG_FILE="configs/conformertcm_baseline.yaml"
    MODEL_SAVE_PATH="out/conformertcm_baseline.pth"
    COMMENT="conformertcm_baseline_training"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py \
        --database_path ${DATABASE_PATH} \
        --protocol_path ${PROTOCOL_PATH} \
        --config ${CONFIG_FILE} \
        --batch_size 32 \
        --num_epochs 100 \
        --max_lr 1e-4 \
        --weight_decay 1e-4 \
        --patience 10 \
        --seed 1234 \
        --model_save_path ${MODEL_SAVE_PATH} \
        --comment ${COMMENT} \
        --algo 0
}

# Function to run augmentation test
run_test_augmentation() {
    print_info "Testing Augmentation Functions..."
    bash test_augmentation.sh
}

# Main script
main() {
    # Get GPU ID
    GPU_ID=${2:-$DEFAULT_GPU}

    # Check if mode is provided as argument
    if [ -n "$1" ]; then
        MODE=$1
    else
        # Show interactive menu
        show_menu
        read -p "Select mode (1-6): " choice

        case $choice in
            1) MODE="curriculum" ;;
            2) MODE="curriculum_random" ;;
            3) MODE="balance" ;;
            4) MODE="baseline" ;;
            5) MODE="test" ;;
            6)
                print_info "Exiting..."
                exit 0
                ;;
            *)
                print_error "Invalid choice!"
                exit 1
                ;;
        esac
    fi

    # Print settings
    echo ""
    print_info "Training Mode: ${MODE}"
    print_info "GPU Device: ${GPU_ID}"
    print_info "Database Path: ${DATABASE_PATH}"
    echo ""

    # Run selected mode
    case $MODE in
        curriculum)
            run_curriculum
            ;;
        curriculum_random)
            run_curriculum_random
            ;;
        balance)
            run_balance
            ;;
        baseline)
            run_baseline
            ;;
        test)
            run_test_augmentation
            ;;
        *)
            print_error "Unknown mode: $MODE"
            echo "Available modes: curriculum, curriculum_random, balance, baseline, test"
            exit 1
            ;;
    esac

    # Check exit status
    if [ $? -eq 0 ]; then
        print_success "Training completed successfully!"
    else
        print_error "Training failed!"
        exit 1
    fi
}

# Run main function
main "$@"
