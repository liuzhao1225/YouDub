# -*- coding: utf-8 -*-

import os
import logging
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

os.environ["MODELSCOPE_CACHE"] = r"./models"

# 设置日志级别和格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoProcessor:
    def __init__(self, model='damo/speech_paraformer_asr-en-16k-vocab4199-pytorch', model_revision="v1.0.1"):
        self.asr_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model=model,
            model_revision=model_revision)
        
        # self.punc_pipeline = pipeline(
        #     task=Tasks.punctuation,
        #     model=r'damo/punc_ct-transformer_cn-en-common-vocab471067-large',
        #     model_revision="v1.0.0")

    def extract_audio_from_video(self, video_path, audio_path):
        logging.info(f'Extracting audio from video {video_path}...')
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        logging.info(f'Audio extracted and saved to {audio_path}.')

    def convert_audio_to_wav(self, audio_path, wav_path):
        logging.info(f'Converting audio {audio_path} to wav...')
        audio = AudioSegment.from_file(audio_path)
        audio.export(wav_path, format="wav")
        logging.info(f'Audio converted and saved to {wav_path}.')

    def transcribe_audio(self, wav_path):
        logging.info(f'Transcribing audio {wav_path}...')
        rec_result = self.asr_pipeline(audio_in=wav_path)
        # rec_result = self.punc_pipeline(text_in=rec_result['text'])
        # rec_result = '\n'.join([f'[{sentence["start"]} - {sentence["end"]}]: {sentence["text"]}' for sentence in rec_result['sentences']])
        rec_result = rec_result['text']
        logging.info('Transcription completed.')
        return rec_result

    def save_transcription_to_txt(self, transcription, txt_path):
        logging.info(f'Saving transcription to {txt_path}...')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        logging.info('Transcription saved.')

    def process_video(self, video_path, audio_path, wav_path, txt_path):
        logging.info('Processing video...')
        self.extract_audio_from_video(video_path, audio_path)
        # self.convert_audio_to_wav(audio_path, wav_path)
        transcription = self.transcribe_audio(audio_path)
        self.save_transcription_to_txt(transcription, txt_path)
        logging.info('Video processing completed.')

# 使用示例
if __name__ == '__main__':
    processor = VideoProcessor()
    processor.process_video('input/Kurzgesagt Channel Trailer.mp4', 'output/audio.mp3', 'output/audio.wav', 'output/transcription.txt')
