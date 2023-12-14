import numpy as np
import librosa
from audiostretchy.stretch import stretch_audio
from scipy.io import wavfile


def adjust_audio_length(wav, src_path, dst_path,  desired_length: float, sample_rate: int = 24000) -> np.ndarray:
    """Adjust the length of the audio.

    Args:
        wav (np.ndarray): Original waveform.
        sample_rate (int): Sampling rate of the audio.
        desired_length (float): Desired length of the audio in seconds.

    Returns:
        np.ndarray: Waveform with adjusted length.
    """
    current_length = wav.shape[0] / sample_rate
    speed_factor = max(min(desired_length / current_length, 1), 0.7)
    desired_length = current_length * speed_factor
    stretch_audio(src_path, dst_path, ratio=speed_factor,
                  sample_rate=sample_rate)
    y, sr = librosa.load(dst_path, sr=sample_rate)
    return y[:int(desired_length * sr)], desired_length


def save_wav(wav: np.ndarray, path: str, sample_rate: int = 24000) -> None:
    """Save float waveform to a file using Scipy.

    Args:
        wav (np.ndarray): Waveform with float values in range [-1, 1] to save.
        path (str): Path to a output file.
        sample_rate (int, optional): Sampling rate used for saving to the file. Defaults to 24000.
    """
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wavfile.write(path, sample_rate, wav_norm.astype(np.int16))

def load_wav(wav_path: str, sample_rate: int = 24000) -> np.ndarray:
    """Load waveform from a file using librosa.

    Args:
        wav_path (str): Path to a file to load.
        sample_rate (int, optional): Sampling rate used for loading the file. Defaults to 24000.

    Returns:
        np.ndarray: Waveform with float values in range [-1, 1].
    """
    return librosa.load(wav_path, sr=sample_rate)[0]