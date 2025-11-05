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

def load_curriculum_augmentation_config(config_path="/home/woongjae/wildspoof/SFM-ADD/aug/augmentation_config_curriculum.yaml"):
    """Load curriculum augmentation configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Global augmentation config (loaded once)
_AUG_CONFIG = None
_CURRICULUM_AUG_CONFIG = None

def get_aug_config(curriculum_stage=None):
    """
    Get augmentation config (lazy loading)

    Args:
        curriculum_stage: If provided (2, 3, or 4), return curriculum config for that stage
                         If None or 1, return default config
    """
    global _AUG_CONFIG, _CURRICULUM_AUG_CONFIG

    if curriculum_stage is None or curriculum_stage == 1:
        # Default config (or clean stage)
        if _AUG_CONFIG is None:
            _AUG_CONFIG = load_augmentation_config()
        return _AUG_CONFIG
    else:
        # Curriculum config
        if _CURRICULUM_AUG_CONFIG is None:
            _CURRICULUM_AUG_CONFIG = load_curriculum_augmentation_config()

        # Return stage-specific config
        stage_key = f'stage{curriculum_stage}'
        if stage_key in _CURRICULUM_AUG_CONFIG:
            return _CURRICULUM_AUG_CONFIG[stage_key]
        else:
            # Fallback to default if stage not found
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

def apply_gaussian_noise(waveform, sr, curriculum_stage=None):
    """Add Gaussian noise to waveform (YAML config)"""
    config = get_aug_config(curriculum_stage)['gaussian_noise']
    std_dev = random.uniform(config['min_std_dev'], config['max_std_dev'])
    noise = np.random.normal(config['mean'], std_dev, len(waveform)).astype(np.float32)
    return waveform + noise


def apply_white_noise(waveform, sr, curriculum_stage=None):
    """Add white noise at random SNR (YAML config)"""
    config = get_aug_config(curriculum_stage).get('white_noise', {})
    min_snr = config.get('min_snr_db', 10)
    max_snr = config.get('max_snr_db', 30)

    # SNR을 min_snr ~ max_snr 사이에서 랜덤 선택
    snr = random.uniform(min_snr, max_snr)
    # 신호 에너지 계산
    signal_power = np.mean(waveform ** 2)
    # SNR에 따른 노이즈 파워 계산
    noise_power = signal_power / (10 ** (snr / 10))
    # White noise 생성
    noise = np.random.normal(0, np.sqrt(noise_power), len(waveform))
    # 신호에 노이즈 추가
    noisy_audio = waveform + noise
    return noisy_audio.astype(np.float32)


def generate_pink_noise(n_samples, sample_rate):
    """
    Generate Pink Noise using FFT with exact length matching.
    """
    # Generate white noise
    white_noise = np.random.randn(n_samples)

    # Compute FFT of white noise
    white_fft = np.fft.rfft(white_noise)

    # Compute frequency bins
    freqs = np.fft.rfftfreq(n_samples, d=1/sample_rate)

    # Compute scaling factors for each frequency bin to create pink noise
    scale = np.ones_like(freqs)
    scale[1:] = 1 / np.sqrt(freqs[1:])  # Exclude DC component

    # Apply scaling to FFT of white noise
    pink_fft = white_fft * scale

    # Inverse FFT to obtain pink noise
    pink_noise = np.fft.irfft(pink_fft, n=n_samples)  # Ensure output length matches input length

    # Normalize Pink Noise to [-1, 1] range
    pink_noise /= np.max(np.abs(pink_noise))

    # Ensure the length of pink noise matches the input signal length
    if len(pink_noise) < n_samples:
        pink_noise = np.pad(pink_noise, (0, n_samples - len(pink_noise)))
    elif len(pink_noise) > n_samples:
        pink_noise = pink_noise[:n_samples]

    return pink_noise


def apply_pink_noise(waveform, sr, curriculum_stage=None):
    """
    Add Pink Noise to an audio signal with a specified SNR (YAML config).
    """
    config = get_aug_config(curriculum_stage).get('pink_noise', {})
    min_snr = config.get('min_snr_db', 10)
    max_snr = config.get('max_snr_db', 30)

    # Generate Pink Noise with the same length as the audio
    pink_noise = generate_pink_noise(len(waveform), sr)

    # Calculate SNR and adjust noise level
    snr = random.uniform(min_snr, max_snr)  # Random SNR from config range
    signal_power = np.mean(waveform ** 2)
    noise_power = signal_power / (10 ** (snr / 10))
    pink_noise *= np.sqrt(noise_power)

    # Add Pink Noise to the audio
    noisy_audio = waveform + pink_noise
    return noisy_audio.astype(np.float32)


def apply_pitch_shift(waveform, sr, curriculum_stage=None):
    """
    Shift pitch by random semitones (YAML config)
    Excludes values near 0 to ensure audible changes
    """
    config = get_aug_config(curriculum_stage)['pitch_shift']
    min_semitones = config['min_semitones']
    max_semitones = config['max_semitones']

    # Randomly choose negative or positive shift (avoiding near-zero changes)
    # e.g., -5 to -2 OR +2 to +5
    if random.random() < 0.5:
        # Negative shift
        n_steps = random.uniform(min_semitones, -2)
    else:
        # Positive shift
        n_steps = random.uniform(2, max_semitones)

    return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)


def apply_time_stretch(waveform, sr, curriculum_stage=None):
    """
    Stretch time by random rate (YAML config)
    Uses discrete values to ensure audible changes
    """
    config = get_aug_config(curriculum_stage)['time_stretch']
    min_factor = config['min_factor']
    max_factor = config['max_factor']

    # Randomly choose slow (< 1.0) or fast (> 1.0) to avoid near-1.0 values
    # e.g., 0.6 to 0.85 OR 1.15 to 1.4
    if random.random() < 0.5:
        # Slow down (rate < 1.0)
        rate = random.uniform(min_factor, 0.85)
    else:
        # Speed up (rate > 1.0)
        rate = random.uniform(1.15, max_factor)

    return librosa.effects.time_stretch(waveform, rate=rate)


def apply_echo(waveform, sr, curriculum_stage=None):
    """Add echo effect (YAML config)"""
    config = get_aug_config(curriculum_stage)['echo']
    delay_sec = random.uniform(config['min_delay'], config['max_delay'])
    decay = random.uniform(config['min_decay'], config['max_decay'])
    delay_samples = int(delay_sec * sr)

    echo = np.zeros(len(waveform) + delay_samples)
    echo[:len(waveform)] = waveform
    echo[delay_samples:delay_samples + len(waveform)] += waveform * decay
    return echo[:len(waveform)].astype(np.float32)


def apply_bandpass_filter(waveform, sr, curriculum_stage=None):
    """Apply random bandpass filter (YAML config for bandpassfilter)"""
    config = get_aug_config(curriculum_stage)['bandpassfilter']
    # Random low and high cutoff frequencies from config ranges
    low = random.uniform(config['min_low_cutoff'], config['max_low_cutoff'])
    high = random.uniform(config['min_high_cutoff'], config['max_high_cutoff'])
    sos = signal.butter(4, [low, high], btype='band', fs=sr, output='sos')
    return signal.sosfilt(sos, waveform).astype(np.float32)


def apply_background_noise(waveform, sr, curriculum_stage=None):
    """Add background noise from actual files (YAML config)"""
    config = get_aug_config(curriculum_stage)['background_noise']
    noise_files = get_noise_files(config['noise_path'])

    min_snr = config.get('snr_db_min', config.get('min_snr_db', 10))
    max_snr = config.get('snr_db_max', config.get('max_snr_db', 30))

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


def apply_background_music(waveform, sr, curriculum_stage=None):
    """Add background music from actual files (YAML config)"""
    config = get_aug_config(curriculum_stage)['background_music']
    music_files = get_music_files(config['music_path'])

    min_snr = config.get('snr_db_min', config.get('min_snr_db', 5))
    max_snr = config.get('snr_db_max', config.get('max_snr_db', 20))

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


def apply_reverberation(waveform, sr, curriculum_stage=None):
    """Apply reverberation using RIR files (YAML config)"""
    config = get_aug_config(curriculum_stage)['reverberation']
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


# def apply_auto_tune(waveform, sr, curriculum_stage=None):
#     """Apply auto-tune (simplified as subtle pitch shift)"""
#     config = get_aug_config(curriculum_stage).get('auto_tune', {})
#     # If strength is specified in config, use it; otherwise default range
#     if 'strength' in config:
#         strength = config['strength']
#         n_steps = random.uniform(-2 * strength, 2 * strength)
#     else:
#         n_steps = random.uniform(-2, 2)
#     return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)


# ===================================================== #
# Augmentation Mapping
# ===================================================== #
NOISE_AUGMENTATION_MAP = {
    # 'auto_tune': apply_auto_tune,  # Commented out - not a true auto-tune implementation
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


def apply_online_augmentation(waveform, sr, noise_type, curriculum_stage=None):
    """
    Apply online augmentation based on noise_type

    Args:
        waveform: numpy array of audio
        sr: sampling rate
        noise_type: target noise type from protocol
        curriculum_stage: Current curriculum stage (1-4) to control augmentation intensity

    Returns:
        augmented waveform
    """
    if noise_type not in NOISE_AUGMENTATION_MAP:
        print(f"[WARNING] Unknown noise type '{noise_type}', returning original")
        return waveform

    try:
        aug_func = NOISE_AUGMENTATION_MAP[noise_type]
        # Pass curriculum_stage to augmentation function
        augmented = aug_func(waveform, sr, curriculum_stage=curriculum_stage)
        return augmented
    except Exception as e:
        print(f"[WARNING] Augmentation failed for {noise_type}: {e}")
        return waveform
