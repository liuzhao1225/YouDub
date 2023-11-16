# -*- coding: utf-8 -*-

import string
import os
import logging
import whisper
import json
from moviepy.editor import VideoFileClip
import sys
sys.path.append(os.getcwd())

from vocal_remover.inference import Vocal_Remover


# 设置日志级别和格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
def merge_segments(transcription, ending=string.punctuation):
    merged_transcription = []
    buffer_segment = None

    for segment in transcription:
        if buffer_segment is None:
            buffer_segment = segment
        else:
            # Check if the last character of the 'text' field is a punctuation mark
            if buffer_segment['text'][-1] in ending:
                # If it is, add the buffered segment to the merged transcription
                merged_transcription.append(buffer_segment)
                buffer_segment = segment
            else:
                # If it's not, merge this segment with the buffered segment
                buffer_segment['text'] += ' ' + segment['text']
                buffer_segment['end'] = segment['end']

    # Don't forget to add the last buffered segment
    if buffer_segment is not None:
        merged_transcription.append(buffer_segment)

    return merged_transcription

class VideoProcessor:
    def __init__(self, model='large', download_root='models/ASR/whisper'):
        logging.info(f'Loading model {model} from {download_root}...')
        self.model = whisper.load_model(model, download_root=download_root)
        self.vocal_remover = Vocal_Remover()
        logging.info('Model loaded.')

    def transcribe_audio(self, wav_path):
        logging.debug(f'Transcribing audio {wav_path}...')
        rec_result = self.model.transcribe(
            wav_path, verbose=True, condition_on_previous_text=False, max_initial_timestamp=None)
        return rec_result
    
    def extract_audio_from_video(self, video_path, audio_path):
        logging.info(f'Extracting audio from video {video_path}...')
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        self.vocal_remover.inference(audio_path, os.path.dirname(audio_path))
        logging.info(f'Audio extracted and saved to {audio_path}.')
    
    

    def save_transcription_to_json(self, transcription, json_path):
        logging.debug(f'Saving transcription to {json_path}...')
        transcription_with_timestemp = [{'start': round(segment['start'], 3), 'end': round(segment['end'], 3), 'text': segment['text'].strip()} for segment in transcription['segments'] if segment['text'] != '']
        
        transcription_with_timestemp = merge_segments(transcription_with_timestemp)
        with open(json_path.replace('en.json', 'subtitle.json'), 'w', encoding='utf-8') as f:
            # f.write(transcription_with_timestemp)
            json.dump(
                transcription_with_timestemp, f, ensure_ascii=False, indent=4)
        
        transcription_with_timestemp = merge_segments(
            transcription_with_timestemp, ending='.?!。？！')
        with open(json_path, 'w', encoding='utf-8') as f:
            # f.write(transcription_with_timestemp)
            json.dump(
                transcription_with_timestemp, f, ensure_ascii=False, indent=8)
            
        logging.debug('Transcription saved.')

    def process_video(self, video_path, output_folder):
        logging.debug('Processing video...')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.extract_audio_from_video(video_path, os.path.join(output_folder, 'en.wav'))
        transcription = self.transcribe_audio(video_path)
        self.save_transcription_to_json(
            transcription, os.path.join(output_folder, 'en.json'))
        logging.debug('Video processing completed.')
        


# 使用示例
if __name__ == '__main__':
    processor = VideoProcessor()
    folder = 'What if you experienced every human life in history_'
    result = processor.transcribe_audio(
        f'output/{folder}/en_Vocals.wav')
    with open(f'output/{folder}/en_without_condition.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    # processor.replace_audio(r'input\Kurzgesagt Channel Trailer.mp4', r'output\Kurzgesagt Channel Trailer\zh.wav',
    # r'output\Kurzgesagt Channel Trailer\zh.json',
    # r'output\Kurzgesagt Channel Trailer\Kurzgesagt Channel Trailer.mp4')

    # with open(r'output\Ancient Life as Old as the Universe\en.json', 'r', encoding='utf-8') as f:
    #     transcript = json.load(f)
        
    # merged_transcript = merge_segments(transcript)
    # print(merged_transcript[:5])
    # with open(r'output\Ancient Life as Old as the Universe\zh.json', 'w', encoding='utf-8') as f:
    #     json.dump(merged_transcript, f, ensure_ascii=False, indent=4)
