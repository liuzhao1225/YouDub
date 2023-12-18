import os
import logging
import json
import re
import time
import numpy as np
from tqdm import tqdm
# from youdub.tts_bytedance import TTS_Clone as TTS_Clone_bytedance, audio_process_folder as audio_process_folder_bytedance
from youdub.tts_xttsv2 import TTS_Clone, audio_process_folder
from youdub.tts_bytedance import TTS_Clone as TTS_Clone_bytedance
from youdub.tts_bytedance import audio_process_folder as audio_process_folder_bytedance
from youdub.asr_whisperX import VideoProcessor
from youdub.video_postprocess import replace_audio_ffmpeg
from youdub.translation_unsafe import Translator
from youdub.utils import split_text
from multiprocessing import Process
import re
import argparse

allowed_chars = '[^a-zA-Z0-9_ .]'


def translate_from_folder(folder, translator: Translator, original_fname):
    with open(os.path.join(folder, 'en.json'), mode='r', encoding='utf-8') as f:
        transcript = json.load(f)
    _transcript = [sentence['text'] for sentence in transcript if sentence['text']]
    result = ['']
    while len(result) != len(_transcript):
        result, summary = translator.translate(_transcript, original_fname)
    for i, sentence in enumerate(result):
        transcript[i]['text'] = sentence
        
    transcript = split_text(transcript) # 使用whisperX后，会自动分句，所以不再需要手动分句。同时避免了将`“你好。”`分为`“你好。`和`”`的情况
    with open(os.path.join(folder, 'zh.json'), 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=4)
    with open(os.path.join(folder, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary)
        
# def main(input_folder, output_folder, diarize=False):
def main():
    parser = argparse.ArgumentParser(description='Process some videos.')
    parser.add_argument('--input_folders', type=str, nargs='+', required=True,
                        help='The list of input folders containing the videos')
    parser.add_argument('--output_folders', type=str, nargs='+', required=True, help='The list of output folders where the processed videos will be stored')
    parser.add_argument('--vocal_only_folders', type=str, nargs='+', default=[],
                        help='The list of input folders containing the videos that only need vocal for the final result.')
    
    parser.add_argument('--diarize', action='store_true',
                        help='Enable diarization')

    
    args = parser.parse_args()

    if len(args.input_folders) != len(args.output_folders):
        raise ValueError(
            "The number of input folders must match the number of output folders.")

    print('='*50)
    print('Initializing...')
    if args.diarize:
        print('Diarization enabled.')
    print('='*50)
    diarize = args.diarize
    processor = VideoProcessor(diarize=diarize)
    translator = Translator()
    tts = TTS_Clone()
    tts_bytedance = TTS_Clone_bytedance()

    for input_folder, output_folder in zip(args.input_folders, args.output_folders):
        if input_folder in args.vocal_only_folders:
            vocal_only = True
            print(f'Vocal only mode enabled for {input_folder}.')
        else:
            vocal_only = False
            
        if not os.path.exists(os.path.join(input_folder, '0_finished')):
            os.makedirs(os.path.join(input_folder, '0_finished'))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if not os.path.exists(os.path.join(output_folder, '0_to_upload')):
            os.makedirs(os.path.join(output_folder, '0_to_upload'))
        if not os.path.exists(os.path.join(output_folder, '0_finished')):
            os.makedirs(os.path.join(output_folder, '0_finished'))
        print('='*50)
        print(
            f'Video processing started for {input_folder} to {output_folder}.')
        print('='*50)

        logging.info('Processing folder...')
        files = os.listdir(input_folder)
        t = tqdm(files, desc="Processing files")
        video_lists = []
        for file in t:
            print('='*50)
            t.set_description(f"Processing {file}")
            print('='*50)
            if file.endswith('.mp4') or file.endswith('.mkv') or file.endswith('.avi') or file.endswith('.flv'):
                original_fname = file[:-4]
                new_filename = re.sub(r'[^a-zA-Z0-9_. ]', '', file)
                new_filename = re.sub(r'\s+', ' ', new_filename)
                new_filename = new_filename.strip()
                os.rename(os.path.join(input_folder, file),
                          os.path.join(input_folder, new_filename))
                file = new_filename
                video_lists.append(file)
                input_path = os.path.join(input_folder, file)
                output_path = os.path.join(output_folder, file[:-4]).strip()
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                speaker_to_voice_type = processor.process_video(
                    input_path, output_path)
            else:
                continue
            if not os.path.exists(os.path.join(output_path, 'zh.json')):
                translate_from_folder(output_path, translator, original_fname)
            if len(speaker_to_voice_type) == 1:
                print('Only one speaker detected. Using TTS.')
                audio_process_folder_bytedance(
                    output_path, tts_bytedance, speaker_to_voice_type, vocal_only=vocal_only)
            else:
                print('Multiple speakers detected. Using XTTSv2.')
                audio_process_folder(
                    output_path, tts)
                
            replace_audio_ffmpeg(os.path.join(input_folder, file), os.path.join(
                output_path, 'zh.wav'),  os.path.join(output_path, 'transcript.json'), os.path.join(output_path, file))
            print('='*50)

        print(
            f'Video processing finished for {input_folder} to {output_folder}. {len(video_lists)} videos processed.')

        print(video_lists)
if __name__ == '__main__':
    # diarize = False
    
    # series = 'TED_Ed'
    # # series = 'z_Others'
    # # series = r'test'
    # # series = 'Kurzgsaget'
    # input_folder = os.path.join(r'input', series)
    # output_folder = os.path.join(r'output', series)
    # main(input_folder, output_folder, diarize=diarize)
    main()