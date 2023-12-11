
import os, sys
sys.path.append(os.getcwd())
from paddlespeech.cli.tts import TTSExecutor
import numpy as np
import json
import logging

from youdub.utils import save_wav, adjust_audio_length



class TTS_Clone:
    def __init__(self, model_path="fastspeech2_male", voc='pwgan_male',device='gpu:0', language='mix'):
        logging.info(f'Loading TTS model {model_path}...')
        self.am = model_path
        self.voc = voc
        self.tts = TTSExecutor()
        self.language = language
        logging.info('Model TTS loaded.')
        
    def inference(self, text, output) -> np.ndarray:
        self.tts(
            text=text,
            am=self.am,
            voc=self.voc,
            lang=self.language,
            output=output,
            use_onnx=True)
        print(f'{output}: {text}')
        
        return self.tts._outputs['wav']

def process_folder(folder, tts: TTS_Clone):
    logging.info(f'TTS processing folder {folder}...')
    with open(os.path.join(folder, 'zh.json'), 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    full_wav = []
    if not os.path.exists(os.path.join(folder, 'temp')):
        os.makedirs(os.path.join(folder, 'temp'))
    
    previous_end = 0
    for i, line in enumerate(transcript):
        text = line['text']
        start = line['start']
        end = line['end']
        
        wav = tts.inference(text, os.path.join(folder, 'temp', f'zh_{i}.wav'))
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
    tts = TTS_Clone()
    process_folder(r'output\Why_The_War_on_Drugs_Is_a_Huge_Failure', tts)
    
    
