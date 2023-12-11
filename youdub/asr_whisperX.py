# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.getcwd())
import whisperx
import whisper
from youdub.demucs_vr import Demucs
import string
import os
import logging
import json
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
load_dotenv()

# 设置日志级别和格式
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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
    def __init__(self, model='large', download_root='models/ASR/whisper', device='cuda', batch_size=32, diarize=False):
        logging.info(f'Loading model {model} from {download_root}...')
        self.device = device
        self.batch_size = batch_size
        self.model = model
        # self.model = whisperx.load_model(model, download_root=download_root, device=device)
        if model == 'large-v3':
            self.whisper_model = whisper.load_model(model, download_root=download_root, device=device) # whisperx doesn't support large-v3 yet, so use whisper instead
        else:
            self.whisper_model = whisperx.load_model(model, download_root=download_root, device=device)
        self.diarize = diarize
        if self.diarize:
            self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=os.getenv('HF_TOKEN'), device=device)

        self.language_code = 'en'
        self.align_model, self.meta_data = whisperx.load_align_model(language_code=self.language_code, device=device)
        self.vocal_remover = Demucs(model='htdemucs_ft')
        logging.info('Model loaded.')

    def transcribe_audio(self, wav_path):
        logging.debug(f'Transcribing audio {wav_path}...')
        if self.model == 'large-v3':
            rec_result = self.whisper_model.transcribe(
                wav_path, verbose=True, condition_on_previous_text=True, max_initial_timestamp=None)
        else:
            rec_result = self.whisper_model.transcribe(
                wav_path, batch_size=self.batch_size, print_progress=True, combined_progress=True)
            
        if rec_result['language'] == 'nn':
            return None
        if rec_result['language'] != self.language_code:
            self.language_code = rec_result['language']
            print(self.language_code)
            self.align_model, self.meta_data = whisperx.load_align_model(language_code=self.language_code, device=self.device)
            
        rec_result = whisperx.align(rec_result['segments'], self.align_model, self.meta_data, wav_path, self.device, return_char_alignments=False, print_progress=True)
        return rec_result
    
    def diarize_transcribed_audio(self, wav_path, transcribe_result):
        logging.info(f'Diarizing audio {wav_path}...')
        diarize_segments = self.diarize_model(wav_path)
        result = whisperx.assign_word_speakers(
            diarize_segments, transcribe_result)
        return result
    
    def extract_audio_from_video(self, video_path, audio_path):
        logging.info(f'Extracting audio from video {video_path}...')
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        output_dir = os.path.dirname(audio_path)
        if not os.path.exists(os.path.join(output_dir, 'en_Vocals.wav')) or not os.path.exists(os.path.join(output_dir, 'en_Instruments.wav')):
            self.vocal_remover.inference(
                audio_path, os.path.dirname(audio_path))
        logging.info(f'Audio extracted and saved to {audio_path}.')

    def save_transcription_to_json(self, transcription, json_path):
        logging.debug(f'Saving transcription to {json_path}...')
        if transcription is None:
            transcription_with_timestemp = []
        else:
            transcription_with_timestemp = [{'start': round(segment['start'], 3), 'end': round(
            segment['end'], 3), 'text': segment['text'].strip(), 'speaker': segment.get('speaker', 'SPEAKER_00')} for segment in transcription['segments'] if segment['text'] != '']

        transcription_with_timestemp = merge_segments(
            transcription_with_timestemp)
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
        if not os.path.exists(os.path.join(output_folder, 'en.wav')):
            self.extract_audio_from_video(video_path, os.path.join(output_folder, 'en.wav'))
        if not os.path.exists(os.path.join(output_folder, 'en.json')):
            transcription = self.transcribe_audio(
                os.path.join(output_folder, 'en_Vocals.wav'))
            if self.diarize:
                transcription = self.diarize_transcribed_audio(
                    os.path.join(output_folder, 'en.wav'), transcription)
            self.save_transcription_to_json(
                transcription, os.path.join(output_folder, 'en.json'))
        logging.debug('Video processing completed.')


# 使用示例
if __name__ == '__main__':
    processor = VideoProcessor(diarize=True)
    folder = r'output\z_Others\Handson with Gemini_ Interacting with multimodal AI'
    # result = processor.transcribe_audio(
    #     f'{folder}/en_Vocals.wav')
    # with open(f'{folder}/en.json', 'w', encoding='utf-8') as f:
    #     json.dump(result, f, ensure_ascii=False, indent=4)
    # result = processor.diarize_transcribed_audio(
    #     f'{folder}/en.wav', result)
    # with open(f'{folder}/en_diarize.json', 'w', encoding='utf-8') as f:
    #     json.dump(result, f, ensure_ascii=False, indent=4)
    processor.process_video(r'input\z_Others\Handson with Gemini_ Interacting with multimodal AI\Handson with Gemini_ Interacting with multimodal AI.mp4', folder)