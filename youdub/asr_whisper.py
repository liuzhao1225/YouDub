# -*- coding: utf-8 -*-

import os
import logging
import whisper
import json
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip,CompositeVideoClip
from moviepy.video.tools.subtitles import SubtitlesClip


# 设置日志级别和格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoProcessor:
    def __init__(self, model='large', download_root='models/ASR/whisper'):
        logging.info(f'Loading model {model} from {download_root}...')
        # self.model = whisper.load_model(model, download_root=download_root)
        self.model = None
        logging.info('Model loaded.')

    def transcribe_audio(self, wav_path):
        logging.debug(f'Transcribing audio {wav_path}...')
        rec_result = self.model.transcribe(wav_path)
        logging.debug('Transcription completed.')
        return rec_result
    
    def extract_audio_from_video(self, video_path, audio_path):
        logging.info(f'Extracting audio from video {video_path}...')
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        logging.info(f'Audio extracted and saved to {audio_path}.')
    
    def replace_audio(self, video_path: str, audio_path: str, subtitle_path: str, output_path: str) -> None:
        """Replace the audio of the video file with the provided audio file.

        Args:
            video_path (str): Path to the video file.
            audio_path (str): Path to the audio file to replace the original audio.
            output_path (str): Path to save the output video file.
        """
        with open(subtitle_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
        
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        def generator(txt): return TextClip(
            txt, fontsize=50, font='STSong', color='white')
        subs = [((segment['start'], segment['end']), segment['text']) for segment in transcript]
        subtitles = SubtitlesClip(subs, generator)
        
        new_video = video.set_audio(audio)
        new_video = CompositeVideoClip(
            [new_video, subtitles.set_position(('center', 'bottom'))])

        new_video.write_videofile(output_path, codec='libx264')

    def save_transcription_to_json(self, transcription, json_path):
        logging.debug(f'Saving transcription to {json_path}...')
        transcription_with_timestemp = [{'start': round(segment['start'], 2), 'end': round(segment['end'], 2), 'text': segment['text'].strip()} for segment in transcription['segments']]
        with open(json_path, 'w', encoding='utf-8') as f:
            # f.write(transcription_with_timestemp)
            json.dump(
                transcription_with_timestemp, f, ensure_ascii=False, indent=4)
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
    processor.replace_audio(r'input\Kurzgesagt Channel Trailer.mp4', r'output\Kurzgesagt Channel Trailer\zh.wav',
    r'output\Kurzgesagt Channel Trailer\zh.json',
    r'output\Kurzgesagt Channel Trailer\Kurzgesagt Channel Trailer.mp4')
