#!/usr/bin/env python3
"""
Augmentation Test Script
Tests all augmentation functions and generates sample outputs for verification.

Usage:
    python test_augmentation.py --input <audio_file> --output_dir <output_directory>

This script will:
1. Load a test audio file
2. Apply each augmentation type for each curriculum stage
3. Save the augmented audio files
4. Generate a report with augmentation statistics
"""

import os
import sys
import argparse
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datautils.noise_augmented import (
    apply_online_augmentation,
    NOISE_AUGMENTATION_MAP,
    load_curriculum_augmentation_config
)


def create_output_dirs(base_dir):
    """Create output directories for each stage"""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    stage_dirs = {}
    for stage in [1, 2, 3, 4]:
        stage_dir = base_path / f"stage{stage}"
        stage_dir.mkdir(exist_ok=True)
        stage_dirs[stage] = stage_dir

    return stage_dirs


def test_single_augmentation(waveform, sr, aug_name, curriculum_stage, output_path):
    """
    Test a single augmentation and save the result

    Args:
        waveform: Input audio waveform
        sr: Sample rate
        aug_name: Name of augmentation (e.g., 'gaussian_noise')
        curriculum_stage: Curriculum stage (1-4)
        output_path: Path to save augmented audio

    Returns:
        dict: Statistics about the augmentation
    """
    try:
        # Apply augmentation
        augmented = apply_online_augmentation(
            waveform.copy(),
            sr,
            aug_name,
            curriculum_stage=curriculum_stage
        )

        # Save augmented audio
        sf.write(output_path, augmented, sr)

        # Calculate statistics
        original_rms = np.sqrt(np.mean(waveform ** 2))
        augmented_rms = np.sqrt(np.mean(augmented ** 2))

        # Calculate SNR if possible (only if lengths match)
        if len(waveform) == len(augmented):
            diff = waveform - augmented
            if np.any(diff != 0):
                noise_power = np.mean(diff ** 2)
                signal_power = np.mean(waveform ** 2)
                if noise_power > 0 and signal_power > 0:
                    snr_db = 10 * np.log10(signal_power / noise_power)
                else:
                    snr_db = float('inf')
            else:
                snr_db = float('inf')
        else:
            # Lengths don't match (e.g., time_stretch) - can't compute SNR this way
            snr_db = 'N/A (length changed)'

        stats = {
            'augmentation': aug_name,
            'stage': curriculum_stage,
            'success': True,
            'original_rms': float(original_rms),
            'augmented_rms': float(augmented_rms),
            'snr_db': snr_db if isinstance(snr_db, str) else (float(snr_db) if snr_db != float('inf') else 'inf'),
            'length_original': len(waveform),
            'length_augmented': len(augmented),
            'output_file': str(output_path)
        }

        return stats

    except Exception as e:
        print(f"  [ERROR] Failed to apply {aug_name} (stage {curriculum_stage}): {e}")
        return {
            'augmentation': aug_name,
            'stage': curriculum_stage,
            'success': False,
            'error': str(e)
        }


def generate_report(all_stats, report_path):
    """Generate a text report with all augmentation results"""
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("AUGMENTATION TEST REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Group by stage
        for stage in [1, 2, 3, 4]:
            stage_stats = [s for s in all_stats if s.get('stage') == stage]

            f.write(f"\n{'=' * 80}\n")
            f.write(f"STAGE {stage}\n")
            f.write(f"{'=' * 80}\n\n")

            if stage == 1:
                f.write("Note: Stage 1 uses clean samples only (no augmentation)\n\n")

            success_count = sum(1 for s in stage_stats if s.get('success'))
            fail_count = len(stage_stats) - success_count

            f.write(f"Total augmentations: {len(stage_stats)}\n")
            f.write(f"Successful: {success_count}\n")
            f.write(f"Failed: {fail_count}\n\n")

            f.write(f"{'Augmentation':<20} {'Status':<10} {'SNR (dB)':<12} {'RMS Ratio':<12} {'Output File'}\n")
            f.write("-" * 120 + "\n")

            for stat in stage_stats:
                aug_name = stat['augmentation']
                if stat['success']:
                    snr = stat.get('snr_db', 'N/A')
                    if snr == 'inf':
                        snr_str = 'N/A'
                    elif isinstance(snr, str):
                        snr_str = snr[:12]  # Truncate if string is too long
                    else:
                        snr_str = f"{snr:.2f}"

                    rms_ratio = stat['augmented_rms'] / stat['original_rms'] if stat['original_rms'] > 0 else 0
                    output_file = Path(stat['output_file']).name

                    f.write(f"{aug_name:<20} {'SUCCESS':<10} {snr_str:<12} {rms_ratio:<12.4f} {output_file}\n")
                else:
                    error = stat.get('error', 'Unknown error')
                    f.write(f"{aug_name:<20} {'FAILED':<10} {error}\n")

            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Test audio augmentation functions')
    parser.add_argument('--input', type=str, required=True,
                        help='Input audio file path')
    parser.add_argument('--output_dir', type=str, default='test_augmentation_output',
                        help='Output directory for augmented files')
    parser.add_argument('--stages', type=str, default='1,2,3,4',
                        help='Comma-separated list of stages to test (default: 1,2,3,4)')
    parser.add_argument('--augmentations', type=str, default='all',
                        help='Comma-separated list of augmentations to test, or "all" (default: all)')

    args = parser.parse_args()

    # Parse stages
    stages = [int(s.strip()) for s in args.stages.split(',')]

    # Parse augmentations
    if args.augmentations.lower() == 'all':
        augmentations = list(NOISE_AUGMENTATION_MAP.keys())
    else:
        augmentations = [a.strip() for a in args.augmentations.split(',')]

    print("=" * 80)
    print("AUGMENTATION TEST SCRIPT")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Testing stages: {stages}")
    print(f"Testing augmentations: {len(augmentations)} types")
    print("=" * 80)
    print()

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        return 1

    # Load audio
    print(f"Loading audio file: {args.input}")
    try:
        waveform, sr = librosa.load(args.input, sr=16000)
        print(f"  Sample rate: {sr} Hz")
        print(f"  Duration: {len(waveform) / sr:.2f} seconds")
        print(f"  Samples: {len(waveform)}")
        print()
    except Exception as e:
        print(f"[ERROR] Failed to load audio: {e}")
        return 1

    # Create output directories
    print(f"Creating output directories in: {args.output_dir}")
    stage_dirs = create_output_dirs(args.output_dir)

    # Save original audio
    original_path = Path(args.output_dir) / "original.wav"
    sf.write(original_path, waveform, sr)
    print(f"  Saved original audio: {original_path}")
    print()

    # Test augmentations
    all_stats = []

    for stage in stages:
        print(f"\n{'=' * 80}")
        print(f"Testing Stage {stage}")
        print(f"{'=' * 80}\n")

        if stage == 1:
            print("Stage 1: Clean only (no augmentation applied)")
            # For stage 1, just copy the original
            output_path = stage_dirs[stage] / "clean.wav"
            sf.write(output_path, waveform, sr)
            all_stats.append({
                'augmentation': 'clean',
                'stage': 1,
                'success': True,
                'original_rms': float(np.sqrt(np.mean(waveform ** 2))),
                'augmented_rms': float(np.sqrt(np.mean(waveform ** 2))),
                'snr_db': 'inf',
                'length_original': len(waveform),
                'length_augmented': len(waveform),
                'output_file': str(output_path)
            })
            print(f"  Saved: {output_path}")
            continue

        for aug_name in augmentations:
            print(f"  Testing: {aug_name} (stage {stage})...", end=' ')

            # Create output filename
            output_filename = f"{aug_name}.wav"
            output_path = stage_dirs[stage] / output_filename

            # Test augmentation
            stats = test_single_augmentation(
                waveform, sr, aug_name, stage, output_path
            )
            all_stats.append(stats)

            if stats['success']:
                snr = stats.get('snr_db', 'N/A')
                if snr == 'inf':
                    print(f"✓ SUCCESS")
                elif isinstance(snr, str):
                    print(f"✓ SUCCESS ({snr})")
                else:
                    print(f"✓ SUCCESS (SNR: {snr:.2f} dB)")
            else:
                print(f"✗ FAILED")

    # Generate report
    print(f"\n{'=' * 80}")
    print("Generating report...")
    report_path = Path(args.output_dir) / "augmentation_report.txt"
    generate_report(all_stats, report_path)
    print(f"  Report saved: {report_path}")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    total_tests = len(all_stats)
    successful = sum(1 for s in all_stats if s.get('success'))
    failed = total_tests - successful

    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful / total_tests * 100:.1f}%")
    print()
    print(f"Output directory: {args.output_dir}")
    print(f"Report file: {report_path}")
    print("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
