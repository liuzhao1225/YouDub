
import os
from TTS.api import TTS
from tqdm import tqdm
import numpy as np
import json
import logging
from .utils import save_wav, adjust_audio_length

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
        
    def inference(self, text, speaker_wav) -> np.ndarray:
        wav = self.tts.tts(
                text=text, speaker_wav=speaker_wav, language=self.language)
        wav = np.array(wav)
        wav /= np.max(np.abs(wav))
        return wav

def process_folder(folder, tts: TTS_Clone):
    logging.info(f'TTS processing folder {folder}...')
    with open(os.path.join(folder, 'zh.json'), 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    full_wav = []
    if not os.path.exists(os.path.join(folder, 'temp')):
        os.makedirs(os.path.join(folder, 'temp'))
    
    previous_end = 0
    for i, line in tqdm(enumerate(transcript)):
        text = line['text']
        start = line['start']
        end = line['end']
        
        wav = tts.inference(text, os.path.join(folder, 'en.wav'))
        save_wav(wav, os.path.join(folder, 'temp', f'zh_{i}.wav'))

        wav_adjusted = adjust_audio_length(wav, os.path.join(folder, 'temp', f'zh_{i}.wav'), os.path.join(
            folder, 'temp',  f'zh_{i}_adjusted.wav'), end - start)
        length = len(wav_adjusted)/24000
        end = start + length
        if start > previous_end:
            full_wav.append(np.zeros(( int(24000 * (start - previous_end)),)))
        full_wav.append(wav_adjusted)
        previous_end = end
    full_wav = np.concatenate(full_wav)
    save_wav(full_wav, os.path.join(folder, f'zh.wav'))
        

if __name__ == '__main__':
    tts = TTS_Clone("tts_models/multilingual/multi-dataset/xtts_v2", language='zh-cn')
    process_folder(r'output\Kurzgesagt Channel Trailer', tts)
    
    
