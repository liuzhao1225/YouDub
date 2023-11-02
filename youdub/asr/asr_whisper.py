# -*- coding: utf-8 -*-

import os
import logging
import whisper
from tqdm import tqdm

# 设置日志级别和格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoProcessor:
    def __init__(self, model='damo/speech_paraformer_asr-en-16k-vocab4199-pytorch', model_revision="v1.0.1"):
        self.model = whisper.load_model("large", download_root="models/whisper")

    def transcribe_audio(self, wav_path):
        logging.info(f'Transcribing audio {wav_path}...')
        rec_result = self.model.transcribe(wav_path)
        rec_result = '\n'.join([f'[{sentence["start"]:.2f} - {sentence["end"]:.2f}]: {sentence["text"]}' for sentence in rec_result['segments']])
        logging.info('Transcription completed.')
        return rec_result

    def save_transcription_to_txt(self, transcription, txt_path):
        logging.info(f'Saving transcription to {txt_path}...')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        logging.info('Transcription saved.')

    def process_video(self, video_path, txt_path):
        logging.info('Processing video...')
        transcription = self.transcribe_audio(video_path)
        self.save_transcription_to_txt(transcription, txt_path)
        logging.info('Video processing completed.')
        
    def process_folder(self, input_folder, output_folder):
        logging.info('Processing folder...')
        files = os.listdir(input_folder)
        for file in tqdm(files):
            if file.endswith('.mp4') or file.endswith('.mkv') or file.endswith('.avi') or file.endswith('.flv'):
                input_path = os.path.join(input_folder, file)
                output_path = os.path.join(output_folder, file[:-4] + '.txt')
                self.process_video(input_path, output_path)
        logging.info('Folder processing completed.')

# 使用示例
if __name__ == '__main__':
    processor = VideoProcessor()
    processor.process_folder('input', 'output')
