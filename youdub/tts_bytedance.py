# coding=utf-8
from datetime import datetime
from requests.exceptions import ConnectionError, Timeout, RequestException
import os
import sys
import time

import numpy as np
sys.path.append(os.getcwd())
from youdub.utils import save_wav, adjust_audio_length
import logging
from loguru import logger
import requests
import uuid
import os
import json

import base64
from dotenv import load_dotenv
load_dotenv()
speaker2voice_name = {
    'SPEAKER_00': 'BV701_streaming',
    'SPEAKER_01': 'BV102_streaming',
    'SPEAKER_02': 'BV002_streaming',
    'SPEAKER_03': 'BV001_streaming',
    'SPEAKER_04': 'BV700_streaming',
    'SPEAKER_05': 'BV007_streaming',
}


def process_tts_input(text):
    text = text.strip()
    return text

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


def process_folder(folder, tts: TTS_Clone):
    logging.info(f'TTS processing folder {folder}...')
    with open(os.path.join(folder, 'zh.json'), 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    full_wav = []
    if not os.path.exists(os.path.join(folder, 'temp')):
        os.makedirs(os.path.join(folder, 'temp'))
    
    previous_end = 0
    for i, line in enumerate(transcript):
        text = line['text']
        start = line['start']
        end = line['end']
        
        wav = tts.inference(text, os.path.join(folder, 'temp', f'zh_{i}.wav'))
        wav_adjusted = adjust_audio_length(wav, os.path.join(folder, 'temp', f'zh_{i}.wav'), os.path.join(
            folder, 'temp',  f'zh_{i}_adjusted.wav'), end - start)
        length = len(wav_adjusted)/24000
        end = start + length
        if start > previous_end:
            full_wav.append(np.zeros(( int(24000 * (start - previous_end)),)))
        full_wav.append(wav_adjusted)
        previous_end = end
    full_wav = np.concatenate(full_wav)
    save_wav(full_wav, os.path.join(folder, f'zh.wav'))

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

