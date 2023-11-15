import os
import logging
import json
import librosa
import numpy as np
from tqdm import tqdm
from youdub.tts_paddle import TTS_Clone
from youdub.asr_whisper import VideoProcessor
from youdub.translation import Translator
from youdub.utils import save_wav, adjust_audio_length
    
def audio_process_folder(folder, tts: TTS_Clone):
    logging.info(f'TTS processing folder {folder}...')
    with open(os.path.join(folder, 'zh.json'), 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    full_wav = np.zeros((0,))
    if not os.path.exists(os.path.join(folder, 'temp')):
        os.makedirs(os.path.join(folder, 'temp'))
        
    for i, line in tqdm(enumerate(transcript)):
        text = line['text']
        # start = line['start']
        start = line['start']
        last_end = len(full_wav)/24000
        if start > last_end:
            full_wav = np.concatenate(
                (full_wav, np.zeros((int(24000 * (start - last_end)),))))
        start = len(full_wav)/24000
        end = line['end']
        wav = tts.inference(text, os.path.join(folder, 'en_Vocals.wav'))
        save_wav(wav, os.path.join(folder, 'temp', f'zh_{i}.wav'))
        wav_adjusted = adjust_audio_length(wav, os.path.join(folder, 'temp', f'zh_{i}.wav'), os.path.join(
            folder, 'temp',  f'zh_{i}_adjusted.wav'), end - start)
        
        
        full_wav = np.concatenate(
            (full_wav, wav_adjusted))
    # load os.path.join(folder, 'en_Instruments.wav')
    # combine with full_wav (the length of the two audio might not be equal)
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
    instruments_wav /= np.max(np.abs(instruments_wav))
    combined_wav = full_wav + instruments_wav
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
            if i - sentence_start < 5:
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
        transcipt = json.load(f)
    _transcript = [sentence['text'] for sentence in transcipt if sentence['text']]
    result = ['']
    while len(result) != len(_transcript):
        result = translator.translate(_transcript)
    for i, sentence in enumerate(result):
        transcipt[i]['text'] = sentence
        
    transcipt = split_text(transcipt)
    with open(os.path.join(folder, 'zh.json'), 'w', encoding='utf-8') as f:
        json.dump(transcipt, f, ensure_ascii=False, indent=4)
    transcipt = split_text(transcipt, punctuations=['，'])
    with open(os.path.join(folder, 'transcript.json'), 'w', encoding='utf-8') as f:
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
    
    logging.info('Processing folder...')
    files = os.listdir(input_folder)
    t = tqdm(files, desc="Processing files")
    video_lists = []
    for file in t:
        print('='*50)
        t.set_description(f"Processing {file}")
        print('='*50)
        if file.endswith('.mp4') or file.endswith('.mkv') or file.endswith('.avi') or file.endswith('.flv'):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file[:-4])
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            processor.process_video(input_path, output_path)
        else:
            continue
        # logging.info(f'Processing {video}...')
        translate_from_folder(output_path, translator)
        audio_process_folder(output_path, tts)
        processor.replace_audio(os.path.join(input_folder, file), os.path.join(
            output_path, 'zh.wav'),  os.path.join(output_path, 'transcript.json'), os.path.join(output_path, file))
if __name__ == '__main__':
    input_folder = r'input'
    input_folder = r'test'
    output_folder = r'output'
    main(input_folder, output_folder)
    
    # with open(r'output\Kurzgesagt Channel Trailer/zh.json', 'r', encoding='utf-8') as f:
    #     transcript = json.load(f)
    # print(split_text(transcript))
