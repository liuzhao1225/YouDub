from demucs.api import Separator
import os
import numpy as np
from scipy.io import wavfile

class Demucs:
    def __init__(self, model="htdemucs_ft", device='cuda', progress=True, shifts=5) -> None:
        print(f'Loading Demucs model {model}...')
        self.separator = Separator(
            model=model, device=device, progress=progress, shifts=shifts)
        print('Demucs model loaded.')

    def inference(self, audio_path: str, output_folder: str) -> None:
        print(f'Demucs separating {audio_path}...')
        origin, separated = self.separator.separate_audio_file(audio_path)
        print(f'Demucs separated {audio_path}.')
        vocals = separated['vocals'].numpy().T
        # vocals.to_file(os.path.join(output_folder, 'en_Vocals.wav'))
        instruments = (separated['drums'] + separated['bass'] + separated['other']).numpy().T
        
        vocal_output_path = os.path.join(output_folder, 'en_Vocals.wav')
        self.save_wav(vocals, vocal_output_path)
        print(f'Demucs saved vocal to {vocal_output_path}.')
        
        instruments_output_path = os.path.join(output_folder, 'en_Instruments.wav')
        self.save_wav(instruments, instruments_output_path)
        print(f'Demucs saved instruments to {instruments_output_path}.')

    def save_wav(self, wav: np.ndarray, output_path:str):
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        wavfile.write(output_path, 44100, wav_norm.astype(np.int16))

if __name__ == '__main__':
    # demucs = Demucs(model='htdemucs_ft')
    demucs = Demucs(model='hdemucs_mmi')
    demucs.inference(r'output\TwoMinutePapers\10000 Of These Train ChatGPT In 4 Minutes\en.wav',
                     r'output\TwoMinutePapers\10000 Of These Train ChatGPT In 4 Minutes')
