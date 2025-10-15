"""
Online Audio Augmentation Functions for Balanced Training

This module provides YAML-config-based audio augmentation functions
for real-time (online) data augmentation during training.
"""

import os
import numpy as np
import librosa
import random
import yaml
import glob
from scipy import signal


# ===================================================== #
# Load Augmentation Config
# ===================================================== #
def load_augmentation_config(config_path="/home/woongjae/wildspoof/SFM-ADD/aug/augmentation_config.yaml"):
    """Load augmentation configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Global augmentation config (loaded once)
_AUG_CONFIG = None

def get_aug_config():
    """Get augmentation config (lazy loading)"""
    global _AUG_CONFIG
    if _AUG_CONFIG is None:
        _AUG_CONFIG = load_augmentation_config()
    return _AUG_CONFIG


# ===================================================== #
# File Caching
# ===================================================== #
_NOISE_FILES_CACHE = {}
_MUSIC_FILES_CACHE = {}
_RIR_FILES_CACHE = {}


def get_noise_files(noise_path):
    """Get list of noise files (cached)"""
    if noise_path not in _NOISE_FILES_CACHE:
        noise_files = glob.glob(os.path.join(noise_path, "**/*.wav"), recursive=True)
        _NOISE_FILES_CACHE[noise_path] = noise_files
        print(f"[INFO] Loaded {len(noise_files)} noise files from {noise_path}")
    return _NOISE_FILES_CACHE[noise_path]


def get_music_files(music_path):
    """Get list of music files (cached)"""
    if music_path not in _MUSIC_FILES_CACHE:
        music_files = glob.glob(os.path.join(music_path, "**/*.wav"), recursive=True)
        _MUSIC_FILES_CACHE[music_path] = music_files
        print(f"[INFO] Loaded {len(music_files)} music files from {music_path}")
    return _MUSIC_FILES_CACHE[music_path]


def get_rir_files(rir_path):
    """Get list of RIR files (cached)"""
    if rir_path not in _RIR_FILES_CACHE:
        rir_files = glob.glob(os.path.join(rir_path, "**/*.wav"), recursive=True)
        _RIR_FILES_CACHE[rir_path] = rir_files
        print(f"[INFO] Loaded {len(rir_files)} RIR files from {rir_path}")
    return _RIR_FILES_CACHE[rir_path]


# ===================================================== #
# Augmentation Functions
# ===================================================== #

def apply_gaussian_noise(waveform, sr):
    """Add Gaussian noise to waveform (YAML config)"""
    config = get_aug_config()['gaussian_noise']
    std_dev = random.uniform(config['min_std_dev'], config['max_std_dev'])
    noise = np.random.normal(config['mean'], std_dev, len(waveform)).astype(np.float32)
    return waveform + noise


def apply_white_noise(waveform, sr):
    """Add white noise at random SNR (YAML config)"""
    config = get_aug_config().get('white_noise', {})
    min_snr = config.get('min_snr_db', 10)
    max_snr = config.get('max_snr_db', 30)

    snr_db = random.uniform(min_snr, max_snr)
    signal_power = np.mean(waveform ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(waveform)) * np.sqrt(noise_power)
    return waveform + noise.astype(np.float32)


def apply_pink_noise(waveform, sr):
    """Add pink noise (1/f noise) at random SNR (YAML config)"""
    config = get_aug_config().get('pink_noise', {})
    min_snr = config.get('min_snr_db', 10)
    max_snr = config.get('max_snr_db', 30)

    snr_db = random.uniform(min_snr, max_snr)

    # Generate white noise
    white = np.random.randn(len(waveform))

    # Apply 1/f filter in frequency domain
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(len(white))
    freqs[0] = 1e-10  # Avoid division by zero
    pink_fft = fft / np.sqrt(freqs)
    pink = np.fft.irfft(pink_fft, n=len(waveform))

    # Normalize and apply SNR
    pink = pink / np.std(pink)
    signal_power = np.mean(waveform ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    pink = pink * np.sqrt(noise_power)

    return waveform + pink.astype(np.float32)


def apply_pitch_shift(waveform, sr):
    """Shift pitch by random semitones (YAML config)"""
    config = get_aug_config()['pitch_shift']
    n_steps = random.uniform(config['min_semitones'], config['max_semitones'])
    return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)


def apply_time_stretch(waveform, sr):
    """Stretch time by random rate (YAML config)"""
    config = get_aug_config()['time_stretch']
    rate = random.uniform(config['min_factor'], config['max_factor'])
    return librosa.effects.time_stretch(waveform, rate=rate)


def apply_echo(waveform, sr):
    """Add echo effect (YAML config)"""
    config = get_aug_config()['echo']
    delay_sec = random.uniform(config['min_delay'], config['max_delay'])
    decay = random.uniform(config['min_decay'], config['max_decay'])
    delay_samples = int(delay_sec * sr)

    echo = np.zeros(len(waveform) + delay_samples)
    echo[:len(waveform)] = waveform
    echo[delay_samples:delay_samples + len(waveform)] += waveform * decay
    return echo[:len(waveform)].astype(np.float32)


def apply_bandpass_filter(waveform, sr):
    """Apply random bandpass filter (YAML config for freq_minus)"""
    config = get_aug_config()['freq_minus']
    # Use config's max_freq as range
    low = random.uniform(100, 500)
    high = random.uniform(3000, config['max_freq'])
    sos = signal.butter(4, [low, high], btype='band', fs=sr, output='sos')
    return signal.sosfilt(sos, waveform).astype(np.float32)


def apply_background_noise(waveform, sr):
    """Add background noise from actual files (YAML config)"""
    config = get_aug_config()['background_noise']
    noise_files = get_noise_files(config['noise_path'])

    min_snr = config.get('min_snr_db', 10)
    max_snr = config.get('max_snr_db', 30)

    if len(noise_files) == 0:
        # Fallback to white noise
        print("[WARNING] No noise files found, using synthetic white noise")
        return apply_white_noise(waveform, sr)

    # Load random noise file
    noise_file = random.choice(noise_files)
    noise, _ = librosa.load(noise_file, sr=sr)

    # Adjust length
    if len(noise) < len(waveform):
        repeats = (len(waveform) // len(noise)) + 1
        noise = np.tile(noise, repeats)

    noise = noise[:len(waveform)]

    # Apply SNR
    snr_db = random.uniform(min_snr, max_snr)
    signal_power = np.mean(waveform ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power > 0:
        noise = noise * np.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))

    return (waveform + noise).astype(np.float32)


def apply_background_music(waveform, sr):
    """Add background music from actual files (YAML config)"""
    config = get_aug_config()['background_music']
    music_files = get_music_files(config['music_path'])

    min_snr = config.get('min_snr_db', 5)
    max_snr = config.get('max_snr_db', 20)

    if len(music_files) == 0:
        # Fallback to white noise
        print("[WARNING] No music files found, using synthetic white noise")
        return apply_white_noise(waveform, sr)

    # Load random music file
    music_file = random.choice(music_files)
    music, _ = librosa.load(music_file, sr=sr)

    # Adjust length
    if len(music) < len(waveform):
        repeats = (len(waveform) // len(music)) + 1
        music = np.tile(music, repeats)

    music = music[:len(waveform)]

    # Apply SNR (music is louder than noise)
    snr_db = random.uniform(min_snr, max_snr)
    signal_power = np.mean(waveform ** 2)
    music_power = np.mean(music ** 2)

    if music_power > 0:
        music = music * np.sqrt(signal_power / (music_power * (10 ** (snr_db / 10))))

    return (waveform + music).astype(np.float32)


def apply_reverberation(waveform, sr):
    """Apply reverberation using RIR files (YAML config)"""
    config = get_aug_config()['reverberation']
    rir_files = get_rir_files(config['rir_path'])

    if len(rir_files) == 0:
        # Fallback to simple reverb
        print("[WARNING] No RIR files found, using simple reverb")
        output = waveform.copy()
        delays = [0.03, 0.05, 0.08, 0.11]
        decays = [0.6, 0.5, 0.4, 0.3]
        for delay, decay in zip(delays, decays):
            delay_samples = int(delay * sr * 0.5)
            if delay_samples < len(waveform):
                shifted = np.zeros(len(waveform))
                shifted[delay_samples:] = waveform[:-delay_samples] * decay
                output += shifted

        max_val = np.max(np.abs(output))
        if max_val > 0:
            return (output / max_val * np.max(np.abs(waveform))).astype(np.float32)
        return output.astype(np.float32)

    # Load random RIR file
    rir_file = random.choice(rir_files)
    rir, _ = librosa.load(rir_file, sr=sr)

    # Apply convolution
    reverb = signal.fftconvolve(waveform, rir, mode='same')

    # Normalize
    max_val = np.max(np.abs(reverb))
    if max_val > 0:
        reverb = reverb / max_val * np.max(np.abs(waveform))

    return reverb.astype(np.float32)


def apply_auto_tune(waveform, sr):
    """Apply auto-tune (simplified as subtle pitch shift)"""
    n_steps = random.uniform(-2, 2)
    return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)


# ===================================================== #
# Augmentation Mapping
# ===================================================== #
NOISE_AUGMENTATION_MAP = {
    'auto_tune': apply_auto_tune,
    'background_music': apply_background_music,
    'background_noise': apply_background_noise,
    'bandpassfilter': apply_bandpass_filter,
    'echo': apply_echo,
    'gaussian_noise': apply_gaussian_noise,
    'pink_noise': apply_pink_noise,
    'pitch_shift': apply_pitch_shift,
    'reverberation': apply_reverberation,
    'time_stretch': apply_time_stretch,
    'white_noise': apply_white_noise,
}


def apply_online_augmentation(waveform, sr, noise_type):
    """
    Apply online augmentation based on noise_type

    Args:
        waveform: numpy array of audio
        sr: sampling rate
        noise_type: target noise type from protocol

    Returns:
        augmented waveform
    """
    if noise_type not in NOISE_AUGMENTATION_MAP:
        print(f"[WARNING] Unknown noise type '{noise_type}', returning original")
        return waveform

    try:
        aug_func = NOISE_AUGMENTATION_MAP[noise_type]
        augmented = aug_func(waveform, sr)
        return augmented
    except Exception as e:
        print(f"[WARNING] Augmentation failed for {noise_type}: {e}")
        return waveform
