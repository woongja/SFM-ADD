from .base import BaseAugmentor
from .utils import librosa_to_pydub
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

class PinkNoiseAugmentor(BaseAugmentor):
    """
    Pink noise augmentation
    Generates pink noise by filtering white noise in frequency domain
    with 1/sqrt(f) weighting.
    
    Config:
    min_std_dev: float, minimum standard deviation (default: 0.01)
    max_std_dev: float, maximum standard deviation (default: 0.2)
    mean: float, mean of noise (default: 0.0)
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.min_std_dev = config.get("min_std_dev", 0.01)
        self.max_std_dev = config.get("max_std_dev", 0.2)
        self.mean = config.get("mean", 0.0)
        self.std_dev = random.uniform(self.min_std_dev, self.max_std_dev)

    def transform(self):
        n = len(self.data)
        # White noise 생성
        white = np.random.normal(self.mean, self.std_dev, n)

        # FFT
        freqs = np.fft.rfftfreq(n, 1.0 / self.sr)
        spectrum = np.fft.rfft(white)

        # 1/sqrt(f) 필터 적용 (0Hz는 무시)
        scale = np.where(freqs == 0, 0, 1.0 / np.sqrt(freqs))
        pink_spectrum = spectrum * scale

        # 역 FFT
        pink = np.fft.irfft(pink_spectrum, n=n).astype(np.float32)

        # 정규화 (에너지 맞추기)
        if np.std(pink) > 0:
            pink = pink / np.std(pink) * self.std_dev

        self.augmented_audio = self.data + pink
        self.augmented_audio = librosa_to_pydub(self.augmented_audio, sr=self.sr)
