
import os, sys
import time
sys.path.append(os.getcwd())
import re
from TTS.api import TTS
import librosa
from tqdm import tqdm
import numpy as np
import json
import logging
from youdub.utils import save_wav, adjust_audio_length, split_text, tts_preprocess_text
from youdub.cn_tx import TextNorm
# Get device
# import torch
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda'

class TTS_Clone:
    def __init__(self, model_path="tts_models/multilingual/multi-dataset/xtts_v2", device='cuda', language='zh-cn'):
        logging.info(f'Loading TTS model {model_path}...')
        self.tts = TTS(model_path).to(device)
        self.language = language
        logging.info('Model TTS loaded.')
        
    def inference(self, text, output_path, speaker_wav) -> np.ndarray:
        wav = self.tts.tts(
                text=text, speaker_wav=speaker_wav, language=self.language)
        wav = np.array(wav)
        save_wav(wav, output_path)
        # wav /= np.max(np.abs(wav))
        return wav





def audio_process_folder(folder, tts: TTS_Clone, speaker_to_voice_type=None, vocal_only=False):
    logging.info(f'TTS processing folder {folder}...')
    logging.info(f'speaker_to_voice_type: {speaker_to_voice_type}')
    with open(os.path.join(folder, 'zh.json'), 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    full_wav = np.zeros((0,))
    if not os.path.exists(os.path.join(folder, 'temp')):
        os.makedirs(os.path.join(folder, 'temp'))

    for i, line in enumerate(transcript):
        text = line['text']
        # start = line['start']
        start = line['start']
        last_end = len(full_wav)/24000
        if start > last_end:
            full_wav = np.concatenate(
                (full_wav, np.zeros((int(24000 * (start - last_end)),))))
        start = len(full_wav)/24000
        line['start'] = start
        end = line['end']
        if os.path.exists(os.path.join(folder, 'temp', f'zh_{str(i).zfill(3)}.wav')):
            wav = librosa.load(os.path.join(
                folder, 'temp', f'zh_{str(i).zfill(3)}.wav'), sr=24000)[0]
        else:
            speaker = line.get('speaker', 'SPEAKER_00')
            speaker_wav = os.path.join(folder, 'SPEAKER', f'{speaker}.wav')
            wav = tts.inference(tts_preprocess_text(text), os.path.join(
                folder, 'temp', f'zh_{str(i).zfill(3)}.wav'), speaker_wav)
            time.sleep(0.1)
        # save_wav(wav, )
        wav_adjusted, adjusted_length = adjust_audio_length(wav, os.path.join(folder, 'temp', f'zh_{str(i).zfill(3)}.wav'), os.path.join(
            folder, 'temp',  f'zh_{str(i).zfill(3)}_adjusted.wav'), end - start)

        wav_adjusted /= wav_adjusted.max()
        line['end'] = line['start'] + adjusted_length
        full_wav = np.concatenate(
            (full_wav, wav_adjusted))
    # load os.path.join(folder, 'en_Instruments.wav')
    # combine with full_wav (the length of the two audio might not be equal)
    transcript = split_text(transcript, punctuations=[
                            '，', '；', '：', '。', '？', '！', '\n'])
    with open(os.path.join(folder, 'transcript.json'), 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=4)
    instruments_wav, sr = librosa.load(
        os.path.join(folder, 'en_Instruments.wav'), sr=24000)

    len_full_wav = len(full_wav)
    len_instruments_wav = len(instruments_wav)

    if len_full_wav > len_instruments_wav:
        # 如果 full_wav 更长，将 instruments_wav 延伸到相同长度
        instruments_wav = np.pad(
            instruments_wav, (0, len_full_wav - len_instruments_wav), mode='constant')
    elif len_instruments_wav > len_full_wav:
        # 如果 instruments_wav 更长，将 full_wav 延伸到相同长度
        full_wav = np.pad(
            full_wav, (0, len_instruments_wav - len_full_wav), mode='constant')
    # 合并两个音频
    full_wav /= np.max(np.abs(full_wav))
    save_wav(full_wav, os.path.join(folder, f'zh_Vocals.wav'))
    # instruments_wav /= np.max(np.abs(instruments_wav))
    instrument_coefficient = 1
    if vocal_only:
        instrument_coefficient = 0
    combined_wav = full_wav + instruments_wav*instrument_coefficient
    combined_wav /= np.max(np.abs(combined_wav))
    save_wav(combined_wav, os.path.join(folder, f'zh.wav'))
        

if __name__ == '__main__':
    folder = r'output\test\Elon Musk on Sam Altman and ChatGPT I am the reason OpenAI exists'
    tts = TTS_Clone("tts_models/multilingual/multi-dataset/xtts_v2", language='zh-cn')
    audio_process_folder(
        folder, tts)
    
    
