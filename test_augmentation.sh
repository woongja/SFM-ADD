#!/bin/bash

# ========================
# Augmentation Test Script
# ========================
# This script tests all augmentation functions and generates sample outputs

# ========================
# Configuration
# ========================

# Input audio file (change this to your test file)
INPUT_FILE="/home/woongjae/wildspoof/Datasets/itw/2.wav"

# Output directory
OUTPUT_DIR="test_augmentation_output"

# Stages to test (1,2,3,4)
STAGES="1,2,3,4"

# Augmentations to test (or "all")
AUGMENTATIONS="all"

# ========================
# Run Test
# ========================

echo "================================================================================"
echo "Testing Audio Augmentation Functions"
echo "================================================================================"
echo ""
echo "Input file: ${INPUT_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Testing stages: ${STAGES}"
echo "Testing augmentations: ${AUGMENTATIONS}"
echo ""
echo "================================================================================"
echo ""

# Check if input file exists
if [ ! -f "${INPUT_FILE}" ]; then
    echo "[ERROR] Input file not found: ${INPUT_FILE}"
    echo ""
    echo "Please edit this script and set INPUT_FILE to a valid audio file path."
    echo "Example paths:"
    echo "  - /home/woongjae/wildspoof/Datasets/bonafide/<filename>.wav"
    echo "  - /home/woongjae/wildspoof/Datasets/spoof/<filename>.wav"
    echo ""
    exit 1
fi

# Run test script
python test_augmentation.py \
    --input "${INPUT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --stages "${STAGES}" \
    --augmentations "${AUGMENTATIONS}"

# Check if test was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "Test completed successfully!"
    echo "================================================================================"
    echo ""
    echo "Output files are saved in: ${OUTPUT_DIR}/"
    echo ""
    echo "Directory structure:"
    echo "  ${OUTPUT_DIR}/"
    echo "  ├── original.wav                  # Original input audio"
    echo "  ├── augmentation_report.txt       # Detailed test report"
    echo "  ├── stage1/                       # Stage 1 samples (clean only)"
    echo "  │   └── clean.wav"
    echo "  ├── stage2/                       # Stage 2 samples (light augmentation)"
    echo "  │   ├── auto_tune.wav"
    echo "  │   ├── background_music.wav"
    echo "  │   ├── background_noise.wav"
    echo "  │   ├── bandpassfilter.wav"
    echo "  │   ├── echo.wav"
    echo "  │   ├── gaussian_noise.wav"
    echo "  │   ├── pink_noise.wav"
    echo "  │   ├── pitch_shift.wav"
    echo "  │   ├── reverberation.wav"
    echo "  │   ├── time_stretch.wav"
    echo "  │   └── white_noise.wav"
    echo "  ├── stage3/                       # Stage 3 samples (medium augmentation)"
    echo "  │   └── (same as stage2)"
    echo "  └── stage4/                       # Stage 4 samples (strong augmentation)"
    echo "      └── (same as stage2)"
    echo ""
    echo "You can listen to the files to verify augmentation quality!"
    echo ""
    echo "To view the detailed report:"
    echo "  cat ${OUTPUT_DIR}/augmentation_report.txt"
    echo ""
else
    echo ""
    echo "================================================================================"
    echo "Test failed!"
    echo "================================================================================"
    echo ""
    echo "Please check the error messages above."
    echo ""
    exit 1
fi
