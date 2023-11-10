import os
import logging
import json
import numpy as np
from tqdm import tqdm
from youdub.asr_whisper import VideoProcessor
from youdub.translation import Translator
from youdub.tts_xttsv2 import TTS_Clone
from youdub.utils import save_wav, adjust_audio_length

def video_process_folder(input_folder, output_folder, processor: VideoProcessor):
    logging.info('Processing folder...')
    files = os.listdir(input_folder)
    t = tqdm(files, desc="Processing files")
    video_lists = []
    for file in t:
        t.set_description(f"Processing {file}")
        if file.endswith('.mp4') or file.endswith('.mkv') or file.endswith('.avi') or file.endswith('.flv'):
            input_path = os.path.join(input_folder, file)
            output_folder = os.path.join(output_folder, file[:-4])
            processor.process_video(input_path, output_folder)
            video_lists.append(file)
    logging.info('Folder processing completed.')
    return video_lists
    
def audio_process_folder(folder, tts: TTS_Clone):
    logging.info(f'TTS processing folder {folder}...')
    with open(os.path.join(folder, 'zh.json'), 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    full_wav = []
    if not os.path.exists(os.path.join(folder, 'temp')):
        os.makedirs(os.path.join(folder, 'temp'))
    for i, line in tqdm(enumerate(transcript)):
        text = line['text']
        start = line['start']
        end = line['end']
        wav = tts.inference(text, os.path.join(folder, 'en.wav'))
        save_wav(wav, os.path.join(folder, 'temp', f'zh_{i}.wav'))

        wav_adjusted = adjust_audio_length(wav, os.path.join(folder, 'temp', f'zh_{i}.wav'), os.path.join(
            folder, 'temp',  f'zh_{i}_adjusted.wav'), end - start)
        full_wav.append(wav_adjusted)
    full_wav = np.concatenate(full_wav)
    save_wav(full_wav, os.path.join(folder, f'zh.wav'))

def translate_from_folder(folder, translator: Translator):
    with open(os.path.join(folder, 'en.json'), mode='r', encoding='utf-8') as f:
        transcipt = json.load(f)
    _transcript = [sentence['text'] for sentence in transcipt]
    result = translator.translate(_transcript)
    for i, sentence in enumerate(result):
        transcipt[i]['text'] = sentence
    with open(os.path.join(folder, 'zh.json'), 'w', encoding='utf-8') as f:
        json.dump(transcipt, f, ensure_ascii=False, indent=4)
        
def main(input_folder, output_folder):
    print('='*50)
    print('Initializing...')
    print('='*50)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    processor = VideoProcessor()
    translator = Translator()
    tts = TTS_Clone()
    
    print('='*50)
    print('Video processing started.')
    print('='*50)
    
    video_lists = video_process_folder(input_folder, output_folder, processor)
    logging.info('\n'.join(video_lists))
    for video in video_lists:
        logging.info(f'Processing {video}...')
        folder = video.split('.')[0]
        folder_path = os.path.join(output_folder, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        translate_from_folder(folder_path, translator)
        audio_process_folder(folder_path, tts)
        processor.replace_audio(os.path.join(input_folder, video), os.path.join(folder_path, 'zh.wav'), os.path.join(folder_path, video))
if __name__ == '__main__':
    input_folder = r'input'
    output_folder = r'output'
    main(input_folder, output_folder)