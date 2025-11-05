import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
from .RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav
from .noise_augmented import apply_online_augmentation
import random

def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    """
    프로토콜 파일을 읽어 파일 경로, 라벨, 노이즈 타입을 반환.
    format: <file_path> <subset> <label> <noise_type>

    For training, also returns clean samples separately for curriculum learning
    """
    d_meta = {}
    file_list = []
    noise_dict = {}
    clean_bonafide_list = []
    clean_spoof_list = []

    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            parts = line.strip().split()
            if len(parts) == 4:
                key, subset, label, noise_type = parts
            elif len(parts) == 3:  # fallback for old format (no noise_type column)
                key, subset, label = parts
                noise_type = "clean"  # Assume all samples are clean if noise_type not provided

            if subset == 'train':
                file_list.append(key)
                d_meta[key] = 1 if label == 'bonafide' else 0
                noise_dict[key] = noise_type

                # Collect clean samples separately for curriculum learning
                if noise_type == 'clean':
                    if label == 'bonafide':
                        clean_bonafide_list.append(key)
                    else:
                        clean_spoof_list.append(key)

        return d_meta, file_list, noise_dict, clean_bonafide_list, clean_spoof_list

    elif is_eval:
        for line in l_meta:
            parts = line.strip().split()
            if len(parts) == 4:
                key, subset, label, noise_type = parts
            elif len(parts) == 3:
                key, subset, label = parts
                noise_type = "clean"  # Assume clean if noise_type not provided

            if subset == 'eval':
                file_list.append(key)
                noise_dict[key] = noise_type

        return file_list

    else:  # dev
        for line in l_meta:
            parts = line.strip().split()
            if len(parts) == 4:
                key, subset, label, noise_type = parts
            elif len(parts) == 3:
                key, subset, label = parts
                noise_type = "clean"  # Assume clean if noise_type not provided

            if subset == 'dev':
                file_list.append(key)
                d_meta[key] = 1 if label == 'bonafide' else 0
                noise_dict[key] = noise_type

        return d_meta, file_list, noise_dict


def pad(
    x: np.ndarray,
    padding_type: str = "zero",
    max_len: int = 64000,
    random_start: bool = True
) -> np.ndarray:
    """
    Pad or crop an audio signal to a fixed length.

    Args:
        x: np.ndarray - input waveform
        padding_type: str - 'zero' or 'repeat'
        max_len: int - output length
        random_start: bool - if True, randomly choose crop start point
    """
    x_len = len(x)
    padded_x = None

    if max_len <= 0:
        raise ValueError("max_len must be >= 0")

    if x_len >= max_len:
        # 길면 자르기 (랜덤 스타트 선택 가능)
        if random_start:
            start = np.random.randint(0, x_len - max_len + 1)
            padded_x = x[start:start + max_len]
        else:
            padded_x = x[:max_len]

    else:
        # 짧으면 패딩 or 반복
        if padding_type == "repeat":
            num_repeats = int(max_len / x_len) + 1
            padded_x = np.tile(x, num_repeats)[:max_len]
        elif padding_type == "zero":
            padded_x = np.zeros(max_len, dtype=x.dtype)
            padded_x[:x_len] = x

    return padded_x


# ===================================================== #
# RawBoost 데이터 증강 (랜덤 적용)
# ===================================================== #
def process_Rawboost_feature(feature, sr, args, algo, prob=0.5, random_algo=False):
    """
    Args:
        feature: waveform
        sr: sampling rate
        args: argument parser object
        algo: 지정된 RawBoost 알고리즘 (1~8)
        prob: 증강을 적용할 확률 (default=0.5)
        random_algo: True면 1~8 중 랜덤 선택
    """

    # ---- 1. 확률적으로 증강 적용 여부 결정 ---- #
    if random.random() > prob or algo == 0:
        return feature  # 그대로 반환 (No augmentation)

    # ---- 2. 알고리즘 랜덤 선택 모드 ---- #
    if random_algo:
        algo = random.randint(1, 8)

    # ---- 3. 알고리즘에 따른 증강 적용 ---- #
    if algo == 1:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin,
            args.maxBiasLinNonLin, sr
        )

    elif algo == 2:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    elif algo == 3:
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands,
            args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr
        )

    elif algo == 4:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin,
            args.maxBiasLinNonLin, sr
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands,
            args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr
        )

    elif algo == 5:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin,
            args.maxBiasLinNonLin, sr
        )
        feature = ISD_additive_noise(feature, args.P, args.g_sd)

    elif algo == 6:
        feature = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin,
            args.maxBiasLinNonLin, sr
        )
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands,
            args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr
        )

    elif algo == 7:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(
            feature, args.SNRmin, args.SNRmax, args.nBands,
            args.minF, args.maxF, args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr
        )

    elif algo == 8:
        feature1 = LnL_convolutive_noise(
            feature, args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG, args.minBiasLinNonLin,
            args.maxBiasLinNonLin, sr
        )
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)
        feature_para = feature1 + feature2
        feature = normWav(feature_para, 0)

    return feature


class Dataset_train(Dataset):
    """
    Curriculum Learning용 Training Dataset

    현재 stage에 맞는 augmentation만 적용
    """
    def __init__(self, args, list_IDs, labels, noise_labels, base_dir, algo,
                 rb_prob=0.5, random_algo=False, random_start=True):
        self.list_IDs = list_IDs
        self.labels = labels
        self.noise_labels = noise_labels
        self.base_dir = base_dir
        self.algo = algo
        self.args = args
        self.cut = 64600
        self.rb_prob = rb_prob
        self.random_algo = random_algo
        self.random_start = random_start

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        wav_path = os.path.join(self.base_dir, utt_id)
        X, fs = librosa.load(wav_path, sr=16000)

        X = process_Rawboost_feature(
            X, fs, self.args, self.algo,
            prob=self.rb_prob, random_algo=self.random_algo
        )

        X_pad = pad(X, max_len=self.cut, random_start=self.random_start)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]
        noise_type = self.noise_labels[utt_id]
        return x_inp, target, noise_type


class Dataset_dev(Dataset):
    """
    Dev set용 Dataset (curriculum learning에서도 동일하게 사용)
    """
    def __init__(self, list_IDs, labels, noise_labels, base_dir):
        self.list_IDs = list_IDs
        self.labels = labels
        self.noise_labels = noise_labels
        self.base_dir = base_dir
        self.cut = 64600

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        wav_path = os.path.join(self.base_dir, utt_id)
        X, fs = librosa.load(wav_path, sr=16000)

        target = self.labels[utt_id]
        original_noise = self.noise_labels[utt_id]

        X_pad = pad(X, max_len=self.cut, random_start=True)
        x_inp = Tensor(X_pad)

        return x_inp, target, original_noise


class Dataset_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        wav_path = os.path.join(self.base_dir, utt_id)
        X, fs = librosa.load(wav_path, sr=16000)
        X_pad = pad(X, max_len=self.cut, random_start=False)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id


def online_augmentation_collate_fn(batch_info, base_dir, args, curriculum_stage=1, log_augmentation=False):
    """
    Custom collate function for online augmentation with curriculum learning

    Args:
        batch_info: List of (utt_id, target_noise_type, label) tuples from sampler
        base_dir: Base directory for audio files
        args: Args containing RawBoost parameters and curriculum stage info
        curriculum_stage: Current curriculum stage (1=clean, 2=light, 3=medium, 4=strong)
        log_augmentation: If True, return augmentation statistics

    Returns:
        batch_x: Tensor of shape [batch_size, audio_length]
        batch_y: Tensor of labels
        batch_noise: List of noise types
        aug_stats: Dict of augmentation statistics (if log_augmentation=True)
    """
    batch_x_list = []
    batch_y_list = []
    batch_noise_list = []
    aug_stats = defaultdict(int)

    for utt_id, target_noise, label in batch_info:
        # Load audio
        wav_path = os.path.join(base_dir, utt_id)
        X, fs = librosa.load(wav_path, sr=16000)

        # Apply online augmentation if not clean
        # Pass curriculum_stage to control augmentation intensity
        if target_noise != 'clean':
            X = apply_online_augmentation(X, fs, target_noise, curriculum_stage=curriculum_stage)
            aug_stats[target_noise] += 1
        else:
            aug_stats['clean'] += 1

        # Apply RawBoost (optional, can be controlled by args)
        if hasattr(args, 'algo') and args.algo > 0:
            X = process_Rawboost_feature(
                X, fs, args, args.algo,
                prob=getattr(args, 'rb_prob', 0.5),
                random_algo=getattr(args, 'rb_random', False)
            )

        # Pad/crop
        X_pad = pad(X, max_len=64600, random_start=True)
        batch_x_list.append(X_pad)
        batch_y_list.append(label)
        batch_noise_list.append(target_noise)

    # Convert to tensors
    batch_x = torch.stack([Tensor(x) for x in batch_x_list])
    batch_y = torch.tensor(batch_y_list, dtype=torch.long)

    if log_augmentation:
        return batch_x, batch_y, batch_noise_list, dict(aug_stats)
    else:
        return batch_x, batch_y, batch_noise_list


class CurriculumNoiseSampler(Sampler):
    """
    Curriculum Learning을 위한 Stage-based Random Noise Sampler

    *** RANDOM SAMPLING VERSION ***
    각 배치는 전체 데이터셋에서 랜덤하게 샘플링 (bonafide/spoof 비율 강제 안 함)

    모든 noise type을 사용하되, stage별로 intensity만 조절:
    - Stage 1: Clean only (5-10 epochs)
    - Stage 2: All noise types with HIGH SNR / Light intensity (5-10 epochs)
    - Stage 3: All noise types with MEDIUM SNR / Medium intensity (5-10 epochs)
    - Stage 4: All noise types with LOW SNR / Strong intensity (5-10 epochs)
    """
    def __init__(self, dataset, noise_labels, label_dict, clean_bonafide_list, clean_spoof_list,
                 base_dir, batch_size=24, curriculum_stage=1):
        self.dataset = dataset
        self.noise_labels = noise_labels
        self.label_dict = label_dict
        self.clean_bonafide_list = clean_bonafide_list
        self.clean_spoof_list = clean_spoof_list
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.curriculum_stage = curriculum_stage

        # Combine bonafide and spoof into a single list with labels
        self.all_samples = []
        for sample_id in clean_bonafide_list:
            self.all_samples.append((sample_id, 1))  # 1 = bonafide
        for sample_id in clean_spoof_list:
            self.all_samples.append((sample_id, 0))  # 0 = spoof

        # All augmentation types (used for stage 2+)
        # Note: auto_tune removed, bandpassfilter covers high_pass and low_pass filters
        self.all_noise_types = [
            'background_music', 'background_noise', 'bandpassfilter',
            'echo', 'gaussian_noise', 'white_noise', 'pink_noise',
            'pitch_shift', 'reverberation', 'time_stretch'
        ]

        # Select augmentation types based on curriculum stage
        if curriculum_stage == 1:
            # Stage 1: Clean only
            self.active_noise_types = []
        else:
            # Stage 2, 3, 4: All noise types, but with different intensity
            self.active_noise_types = self.all_noise_types.copy()

        print(f"\n[CurriculumNoiseSampler - RANDOM SAMPLING - Stage {curriculum_stage}]")
        print(f"  Total samples: {len(self.all_samples)}")
        print(f"    - Bonafide: {len(clean_bonafide_list)}")
        print(f"    - Spoof: {len(clean_spoof_list)}")
        if curriculum_stage == 1:
            print(f"  Mode: Clean only")
        else:
            print(f"  Mode: All noise types with stage {curriculum_stage} intensity")
        print(f"  Active augmentation types: {len(self.active_noise_types)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Sampling strategy: RANDOM (not balanced)")

    def __iter__(self):
        num_batches = len(self)
        # Shuffle all samples at the beginning of each epoch
        shuffled_samples = self.all_samples.copy()
        random.shuffle(shuffled_samples)

        for batch_idx in range(num_batches):
            batch_indices = []
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(shuffled_samples))

            for idx in range(start_idx, end_idx):
                sample_id, label = shuffled_samples[idx]

                if self.curriculum_stage == 1:
                    # Stage 1: Clean only
                    noise_type = 'clean'
                else:
                    # Stage 2+: Randomly choose noise type or clean
                    noise_type = random.choice(self.active_noise_types + ['clean'])

                batch_indices.append((sample_id, noise_type, label))

            yield batch_indices

    def __len__(self):
        # Number of batches
        return len(self.all_samples) // self.batch_size
