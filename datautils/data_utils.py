import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from .RawBoost import ISD_additive_noise, LnL_convolutive_noise, SSI_additive_noise, normWav
import random
SUPPORTED_DATALOADERS = ["data_utils"]
def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            key, subset, label = line.strip().split()
            if subset == 'train':
                file_list.append(key)
                d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            key, subset, label = line.strip().split()
            if subset == 'eval':
                file_list.append(key)
        return file_list

    else:  # dev
        for line in l_meta:
            key, subset, label = line.strip().split()
            if subset == 'dev':
                file_list.append(key)
                d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
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
    def __init__(self, args, list_IDs, labels, base_dir, algo, rb_prob=0.5, random_algo=False):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo = algo
        self.args = args
        self.cut = 64600
        self.rb_prob = rb_prob  # RawBoost 확률
        self.random_algo = random_algo  # 알고리즘 무작위 선택 여부

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        wav_path = os.path.join(self.base_dir, utt_id)
        X, fs = librosa.load(wav_path, sr=16000)

        # 랜덤 RawBoost 적용
        X = process_Rawboost_feature(
            X, fs, self.args, self.algo,
            prob=self.rb_prob, random_algo=self.random_algo
        )

        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        target = self.labels[utt_id]
        return x_inp, target


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
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id
