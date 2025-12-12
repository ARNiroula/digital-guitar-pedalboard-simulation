import numpy as np


def simple_distortion(audio):
    """
    A very simple distortion effect
    """
    return np.clip(audio * 3, -0.8, 0.8)


def distortion(audio, gain):
    pass
