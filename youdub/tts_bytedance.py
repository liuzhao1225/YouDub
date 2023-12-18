# coding=utf-8
from datetime import datetime
import librosa
from requests.exceptions import ConnectionError, Timeout, RequestException
import os
import sys
import time

import numpy as np
sys.path.append(os.getcwd())
from youdub.utils import save_wav, adjust_audio_length, split_text, tts_preprocess_text
import logging
from loguru import logger
import requests
import uuid
import os
import json
import re

import base64
from dotenv import load_dotenv
load_dotenv()

# [700, 705, 701, 001, 406, 407, 002, 701, 123, 120, 119, 115, 107, 100, 104, 004, 113, 102, 405]
    
class TTS_Clone:
    def __init__(self):
        self.appid = os.getenv('APPID')
        self.access_token = os.getenv('ACCESS_TOKEN')
        self.cluster = "volcano_tts"
        self.host = "openspeech.bytedance.com"
        self.api_url = f"https://{self.host}/api/v1/tts"
        self.header = {"Authorization": f"Bearer;{self.access_token}"}
        self.request_json = {
            "app": {
                "appid": self.appid,
                "token": "access_token",
                "cluster": self.cluster
            },
            "user": {
                "uid": "388808087185088"
            },
            "audio": {
                "voice_type": '',
                "encoding": "wav",
                "speed_ratio": 1,
                "volume_ratio": 1.0,
                "pitch_ratio": 1.0,
            },
            "request": {
                "reqid": str(uuid.uuid4()),
                "text": "字节跳动语音合成",
                "text_type": "plain",
                "operation": "query",
                "with_frontend": 1,
                "frontend_type": "unitTson"

            }
        }
        self.output_path = r'.'
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

    def inference(self, text, output_wav_path, speaker='SPEAKER_00', speaker_to_voice_type={'SPEAKER_00': 'BV701_streaming'}):
        self.request_json['request']['text'] = text
        self.request_json['request']['reqid'] = str(uuid.uuid4())
        self.request_json['audio']['voice_type'] = speaker_to_voice_type.get(
                speaker, 'BV701_streaming')
        max_retries = 5
        timeout_seconds = 10  # Set your desired timeout in seconds

        for attempt in range(max_retries):
            try:
                resp = requests.post(self.api_url, json.dumps(
                    self.request_json), headers=self.header, timeout=timeout_seconds)
                if resp.status_code == 200:
                    data = resp.json()["data"]
                    data = base64.b64decode(data)
                    with open(output_wav_path, "wb") as f:
                        f.write(data)
                    print(f'{output_wav_path}: {text}')
                    return np.frombuffer(data, dtype=np.int16)
                else:
                    print(f"Request failed with status code: {resp.status_code}")
                    if resp.status_code == 500:
                        return None
                    raise Exception(f"Request failed with status code: {resp.status_code}")
            except Exception as e:
                print(f"Request failed: {e}, retrying ({attempt+1}/{max_retries})")
                time.sleep(2)  # Wait 2 seconds before retrying

        print("Max retries reached, request failed")
        return None


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
            wav = tts.inference(tts_preprocess_text(text), os.path.join(
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
    transcript = split_text(transcript, punctuations=[
                            '，', '；', '：', '。', '？', '！', '\n'])
    with open(os.path.join(folder, 'transcript.json'), 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=4)
    instruments_wav, sr = librosa.load(
        os.path.join(folder, 'en_Instruments.wav'), sr=24000)

    len_full_wav = len(full_wav)
    len_instruments_wav = len(instruments_wav)

    if len_full_wav > len_instruments_wav:
        # 如果 full_wav 更长，将 instruments_wav 延伸到相同长度
        instruments_wav = np.pad(
            instruments_wav, (0, len_full_wav - len_instruments_wav), mode='constant')
    elif len_instruments_wav > len_full_wav:
        # 如果 instruments_wav 更长，将 full_wav 延伸到相同长度
        full_wav = np.pad(
            full_wav, (0, len_instruments_wav - len_full_wav), mode='constant')
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


if __name__ == '__main__':
    tts = TTS_Clone()
    # process_folder(
    #     r'output\test\Blood concrete and dynamite Building the Hoover Dam Alex Gendler', tts)
    from tqdm import tqdm
    voice_type_folder = r'voice_type'
    voice_type_lst = []
    for fname in os.listdir(voice_type_folder):
        voice_type_lst.append(fname.split('.')[0])
    for voice_type in tqdm(voice_type_lst):
        # voice_type = f'BV{str(i).zfill(3)}_streaming'
        output_wav = f'voice_type/{voice_type}.wav'
        # if os.path.exists(output_wav):
        #     continue
        try:
            tts.inference(
                'YouDub 是一个创新的开源工具，专注于将 YouTube 等平台的优质视频翻译和配音为中文版本。此工具融合了先进的 AI 技术，包括语音识别、大型语言模型翻译以及 AI 声音克隆技术，为中文用户提供具有原始 YouTuber 音色的中文配音视频。更多示例和信息，欢迎访问我的bilibili视频主页。你也可以加入我们的微信群，扫描下方的二维码即可。', output_wav, voice_type=voice_type)
        except:
            print(f'voice {voice_type} failed.')
        time.sleep(0.1)

