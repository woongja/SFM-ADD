from .base import BaseAugmentor
from .utils import librosa_to_pydub
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

class WhiteNoiseAugmentor(BaseAugmentor):
    """
    White noise augmentation
    Adds white noise with mean 0 and std_dev randomly chosen.
    
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
        noise = np.random.normal(self.mean, self.std_dev, len(self.data)).astype(np.float32)
        self.augmented_audio = self.data + noise
        self.augmented_audio = librosa_to_pydub(self.augmented_audio, sr=self.sr)
