# -*- coding: utf-8 -*-

import string
import os
import logging
import whisper
import json
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip,CompositeVideoClip
from moviepy.video.tools.subtitles import SubtitlesClip
import sys
sys.path.append(os.getcwd())

from vocal_remover.inference import Vocal_Remover


# 设置日志级别和格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def format_timestamp(seconds):
    """Converts seconds to the SRT time format."""
    millisec = int((seconds - int(seconds)) * 1000)
    hours, seconds = divmod(int(seconds), 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millisec:03}"

def convert_json_to_srt(json_file, srt_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        subtitles = json.load(f)

    with open(srt_file, 'w', encoding='utf-8') as f:
        for i, subtitle in enumerate(subtitles, 1):
            start = format_timestamp(subtitle['start'])
            end = format_timestamp(subtitle['end'])
            text = subtitle['text']

            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")
            
def replace_audio(video_path: str, audio_path: str, subtitle_path: str, output_path: str, fontsize=64, font='SimHei', color='white') -> None:
    """Replace the audio of the video file with the provided audio file.

    Args:
        video_path (str): Path to the video file.
        audio_path (str): Path to the audio file to replace the original audio.
        output_path (str): Path to save the output video file.
    """
    convert_json_to_srt(subtitle_path, video_path.replace('.mp4', '.srt').replace('.mkv', '.srt').replace('.avi', '.srt').replace('.flv', '.srt'))

    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    new_video = video.set_audio(audio)

    # size = video.size
    # size = (size[0], size[1] * 0.8)
    # def generator(txt): return TextClip(
    #     txt, fontsize=size[0]//20, font=font, color=color, method='caption', align='south', size=size, stroke_color='black', stroke_width=max(fontsize//20, 2))
    # subs = [((segment['start'], segment['end']), segment['text'])
    #         for segment in transcript]
    # subtitles = SubtitlesClip(subs, generator)
    # new_video = CompositeVideoClip(
    #     [new_video, subtitles.set_position(('center', 'bottom'))])

    new_video.write_videofile(output_path, codec='libx264', threads=16)
    
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
