import os
import logging
import json
import re
import time
import librosa
import numpy as np
from tqdm import tqdm
from youdub.tts_bytedance import TTS_Clone
from youdub.asr_whisperX import VideoProcessor
from youdub.video_postprocess import replace_audio_ffmpeg
from youdub.translation_unsafe import Translator
from youdub.utils import save_wav, adjust_audio_length
from multiprocessing import Process
import re
import argparse

allowed_chars = '[^a-zA-Z0-9_ .]'


def add_space_before_capitals(text):
    # 使用正则表达式查找所有的大写字母，并在它们前面加上空格
    # 正则表达式说明：(?<!^) 表示如果不是字符串开头，则匹配，[A-Z] 匹配任何大写字母
    text = re.sub(r'(?<!^)([A-Z])', r' \1', text)
    # 使用正则表达式在字母和数字之间插入空格
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
    return text


def audio_process_folder(folder, tts: TTS_Clone, speaker_to_voice_type, vocal_only=False):
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
            wav = tts.inference(add_space_before_capitals(text), os.path.join(
                folder, 'temp', f'zh_{str(i).zfill(3)}.wav'), speaker=line.get('speaker', 'SPEAKER_00'), speaker_to_voice_type=speaker_to_voice_type)
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
    transcript = split_text(transcript, punctuations=['，', '；', '：'])
    with open(os.path.join(folder, 'transcript.json'), 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=4)
    instruments_wav, sr = librosa.load(
        os.path.join(folder, 'en_Instruments.wav'), sr=24000)
    
    len_full_wav = len(full_wav)
    len_instruments_wav = len(instruments_wav)
    
    if len_full_wav > len_instruments_wav:
        # 如果 full_wav 更长，将 instruments_wav 延伸到相同长度
        instruments_wav = np.pad(instruments_wav, (0, len_full_wav - len_instruments_wav), mode='constant')
    elif len_instruments_wav > len_full_wav:
        # 如果 instruments_wav 更长，将 full_wav 延伸到相同长度
        full_wav = np.pad(full_wav, (0, len_instruments_wav - len_full_wav), mode='constant')
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


def split_text(input_data,
               punctuations=['。', '？', '！', '\n']):
    # Chinese punctuation marks for sentence ending

    # Function to check if a character is a Chinese ending punctuation
    def is_punctuation(char):
        return char in punctuations

    # Process each item in the input data
    output_data = []
    for item in input_data:
        start = item["start"]
        text = item["text"]
        sentence_start = 0
        
        # Calculate the duration for each character
        duration_per_char = (item["end"] - item["start"]) / len(text)
        for i, char in enumerate(text):
            # If the character is a punctuation, split the sentence
            if not is_punctuation(char) and i != len(text) - 1:
                continue
            if i - sentence_start < 5 and i != len(text) - 1:
                    continue
            sentence = text[sentence_start:i+1]
            sentence_end = start + duration_per_char * len(sentence)

            # Append the new item
            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": sentence
            })

            # Update the start for the next sentence
            start = sentence_end
            sentence_start = i + 1

    return output_data

def translate_from_folder(folder, translator: Translator):
    with open(os.path.join(folder, 'en.json'), mode='r', encoding='utf-8') as f:
        transcript = json.load(f)
    _transcript = [sentence['text'] for sentence in transcript if sentence['text']]
    result = ['']
    while len(result) != len(_transcript):
        result = translator.translate(_transcript)
    for i, sentence in enumerate(result):
        transcript[i]['text'] = sentence
        
    # transcript = split_text(transcript) # 使用whisperX后，会自动分句，所以不再需要手动分句。同时避免了将`“你好。”`分为`“你好。`和`”`的情况
    with open(os.path.join(folder, 'zh.json'), 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=4)
    
        
# def main(input_folder, output_folder, diarize=False):
def main():
    parser = argparse.ArgumentParser(description='Process some videos.')
    parser.add_argument('--input_folders', type=str, nargs='+', required=True,
                        help='The list of input folders containing the videos')
    parser.add_argument('--output_folders', type=str, nargs='+', required=True, help='The list of output folders where the processed videos will be stored')
    parser.add_argument('--vocal_only_folders', type=str, nargs='+',
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
                translate_from_folder(output_path, translator)
            audio_process_folder(
                output_path, tts, speaker_to_voice_type, vocal_only=vocal_only)
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