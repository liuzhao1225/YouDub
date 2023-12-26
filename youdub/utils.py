import re
import numpy as np
import librosa
from audiostretchy.stretch import stretch_audio
from scipy.io import wavfile
from youdub.cn_tx import TextNorm
normalizer = TextNorm()

def tts_preprocess_text(text):
    # 使用正则表达式查找所有的大写字母，并在它们前面加上空格
    # 正则表达式说明：(?<!^) 表示如果不是字符串开头，则匹配，[A-Z] 匹配任何大写字母
    text = text.replace('AI', '人工智能')
    text = re.sub(r'(?<!^)([A-Z])', r' \1', text)
    text = normalizer(text)
    # 使用正则表达式在字母和数字之间插入空格
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
    return text

def split_text(input_data,
               punctuations=['。', '？', '！', '\n', "”"]):
    # Chinese punctuation marks for sentence ending

    # Function to check if a character is a Chinese ending punctuation
    def is_punctuation(char):
        return char in punctuations

    # Process each item in the input data
    output_data = []
    for item in input_data:
        start = item["start"]
        text = item["text"]
        speaker = item.get("speaker", "SPEAKER_00")
        sentence_start = 0

        # Calculate the duration for each character
        duration_per_char = (item["end"] - item["start"]) / len(text)
        for i, char in enumerate(text):
            # If the character is a punctuation, split the sentence
            if not is_punctuation(char) and i != len(text) - 1:
                continue
            if i - sentence_start < 5 and i != len(text) - 1:
                continue
            if i < len(text) - 1 and is_punctuation(text[i+1]):
                continue
            sentence = text[sentence_start:i+1]
            sentence_end = start + duration_per_char * len(sentence)

            # Append the new item
            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": sentence,
                "speaker": speaker
            })

            # Update the start for the next sentence
            start = sentence_end
            sentence_start = i + 1

    return output_data

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
    speed_factor = max(min(desired_length / current_length, 1.1), 2/3)
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
    # wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wav_norm = wav * 32767
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