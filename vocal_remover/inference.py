import argparse
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from .lib import dataset
from .lib import nets
from .lib import spec_utils
from .lib import utils


class Separator(object):

    def __init__(self, model, device, batchsize, cropsize, postprocess=False):
        self.model = model
        self.offset = model.offset
        self.device = device
        self.batchsize = batchsize
        self.cropsize = cropsize
        self.postprocess = postprocess

    def _separate(self, X_mag_pad, roi_size):
        X_dataset = []
        patches = (X_mag_pad.shape[2] - 2 * self.offset) // roi_size
        for i in range(patches):
            start = i * roi_size
            X_mag_crop = X_mag_pad[:, :, start:start + self.cropsize]
            X_dataset.append(X_mag_crop)

        X_dataset = np.asarray(X_dataset)

        self.model.eval()
        with torch.no_grad():
            mask = []
            # To reduce the overhead, dataloader is not used.
            for i in tqdm(range(0, patches, self.batchsize)):
                X_batch = X_dataset[i: i + self.batchsize]
                X_batch = torch.from_numpy(X_batch).to(self.device)

                pred = self.model.predict_mask(X_batch)

                pred = pred.detach().cpu().numpy()
                pred = np.concatenate(pred, axis=2)
                mask.append(pred)

            mask = np.concatenate(mask, axis=2)

        return mask

    def _preprocess(self, X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        return X_mag, X_phase

    def _postprocess(self, mask, X_mag, X_phase):
        if self.postprocess:
            mask = spec_utils.merge_artifacts(mask)

        y_spec = mask * X_mag * np.exp(1.j * X_phase)
        v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)

        return y_spec, v_spec

    def separate(self, X_spec):
        X_mag, X_phase = self._preprocess(X_spec)

        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()

        mask = self._separate(X_mag_pad, roi_size)
        mask = mask[:, :, :n_frame]

        y_spec, v_spec = self._postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec

    def separate_tta(self, X_spec):
        X_mag, X_phase = self._preprocess(X_spec)

        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()

        mask = self._separate(X_mag_pad, roi_size)

        pad_l += roi_size // 2
        pad_r += roi_size // 2
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()

        mask_tta = self._separate(X_mag_pad, roi_size)
        mask_tta = mask_tta[:, :, roi_size // 2:]
        mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5

        y_spec, v_spec = self._postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec

# 其他代码保持不变
class Vocal_Remover:
    def __init__(self, pretrained_model='models/VR/baseline.pth', sr=44100, n_fft=2048, hop_length=1024, batchsize=4, cropsize=256, postprocess=False, tta=False, output_image=False, gpu=0):
        class Args:
            def __init__(self):
                self.input = None
                self.output_dir = None
                self.pretrained_model = pretrained_model
                self.sr = sr
                self.n_fft = n_fft
                self.hop_length = hop_length
                self.batchsize = batchsize
                self.cropsize = cropsize
                self.postprocess = postprocess
                self.tta = tta
                self.output_image = output_image
                self.gpu = gpu
        self.args = Args()

        print('loading model...', end=' ')
        self.device = torch.device('cpu')
        self.model = nets.CascadedNet(self.args.n_fft, 32, 128)
        self.model.load_state_dict(torch.load(
            self.args.pretrained_model, map_location=self.device))
        if self.args.gpu >= 0:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:{}'.format(self.args.gpu))
                self.model.to(self.device)
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device('mps')
                self.model.to(self.device)
        print('done')

    def inference(self, input_file, output_dir):
        self.args.input = input_file
        self.args.output_dir = output_dir
        print('loading wave source...', end=' ')
        X, sr = librosa.load(
            self.args.input, sr=self.args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')
        basename = os.path.splitext(os.path.basename(self.args.input))[0]
        print('done')

        if X.ndim == 1:
            # mono to stereo
            X = np.asarray([X, X])

        print('stft of wave source...', end=' ')
        X_spec = spec_utils.wave_to_spectrogram(
            X, self.args.hop_length, self.args.n_fft)
        print('done')

        sp = Separator(self.model, self.device, self.args.batchsize,
                       self.args.cropsize, self.args.postprocess)

        if self.args.tta:
            y_spec, v_spec = sp.separate_tta(X_spec)
        else:
            y_spec, v_spec = sp.separate(X_spec)

        print('validating output directory...', end=' ')
        output_dir = self.args.output_dir
        if output_dir != "":  # modifies output_dir if theres an arg specified
            output_dir = output_dir.rstrip('/') + '/'
            os.makedirs(output_dir, exist_ok=True)
        print('done')

        print('inverse stft of instruments...', end=' ')
        wave = spec_utils.spectrogram_to_wave(
            y_spec, hop_length=self.args.hop_length)
        print('done')
        sf.write('{}{}_Instruments.wav'.format(output_dir, basename), wave.T, sr)

        print('inverse stft of vocals...', end=' ')
        wave = spec_utils.spectrogram_to_wave(
            v_spec, hop_length=self.args.hop_length)
        print('done')
        sf.write('{}{}_Vocals.wav'.format(output_dir, basename), wave.T, sr)

        if self.args.output_image:
            image = spec_utils.spectrogram_to_image(y_spec)
            utils.imwrite('{}{}_Instruments.jpg'.format(
                output_dir, basename), image)

            image = spec_utils.spectrogram_to_image(v_spec)
            utils.imwrite('{}{}_Vocals.jpg'.format(output_dir, basename), image)
    
    
def inference(input_file, output_dir, pretrained_model='models/VR/baseline.pth', sr=44100, n_fft=2048, hop_length=1024, batchsize=4, cropsize=256, postprocess=False, tta=False, output_image=False, gpu=0):
    # 模拟命令行参数
    
    class Args:
        def __init__(self):
            self.input = input_file
            self.output_dir = output_dir
            self.pretrained_model = pretrained_model
            self.sr = sr
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.batchsize = batchsize
            self.cropsize = cropsize
            self.postprocess = postprocess
            self.tta = tta
            self.output_image = output_image
            self.gpu = gpu
    args = Args()

    print('loading model...', end=' ')
    device = torch.device('cpu')
    model = nets.CascadedNet(args.n_fft, 32, 128)
    model.load_state_dict(torch.load(
        args.pretrained_model, map_location=device))
    if args.gpu >= 0:
        if torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(args.gpu))
            model.to(device)
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
            model.to(device)
    print('done')

    print('loading wave source...', end=' ')
    X, sr = librosa.load(
        args.input, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')
    basename = os.path.splitext(os.path.basename(args.input))[0]
    print('done')

    if X.ndim == 1:
        # mono to stereo
        X = np.asarray([X, X])

    print('stft of wave source...', end=' ')
    X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
    print('done')

    sp = Separator(model, device, args.batchsize,
                   args.cropsize, args.postprocess)

    if args.tta:
        y_spec, v_spec = sp.separate_tta(X_spec)
    else:
        y_spec, v_spec = sp.separate(X_spec)

    print('validating output directory...', end=' ')
    output_dir = args.output_dir
    if output_dir != "":  # modifies output_dir if theres an arg specified
        output_dir = output_dir.rstrip('/') + '/'
        os.makedirs(output_dir, exist_ok=True)
    print('done')

    print('inverse stft of instruments...', end=' ')
    wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
    print('done')
    sf.write('{}{}_Instruments.wav'.format(output_dir, basename), wave.T, sr)

    print('inverse stft of vocals...', end=' ')
    wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
    print('done')
    sf.write('{}{}_Vocals.wav'.format(output_dir, basename), wave.T, sr)

    if args.output_image:
        image = spec_utils.spectrogram_to_image(y_spec)
        utils.imwrite('{}{}_Instruments.jpg'.format(
            output_dir, basename), image)

        image = spec_utils.spectrogram_to_image(v_spec)
        utils.imwrite('{}{}_Vocals.jpg'.format(output_dir, basename), image)

# 原来的 if __name__ == '__main__': 块可以被移除或注释掉

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_model', '-P', type=str, default='models/baseline.pth')
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--batchsize', '-B', type=int, default=4)
    p.add_argument('--cropsize', '-c', type=int, default=256)
    p.add_argument('--output_image', '-I', action='store_true')
    p.add_argument('--postprocess', '-p', action='store_true')
    p.add_argument('--tta', '-t', action='store_true')
    p.add_argument('--output_dir', '-o', type=str, default="")
    args = p.parse_args()

    print('loading model...', end=' ')
    device = torch.device('cpu')
    model = nets.CascadedNet(args.n_fft, 32, 128)
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    if args.gpu >= 0:
        if torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(args.gpu))
            model.to(device)
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device('mps')
            model.to(device)
    print('done')

    print('loading wave source...', end=' ')
    X, sr = librosa.load(
        args.input, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')
    basename = os.path.splitext(os.path.basename(args.input))[0]
    print('done')

    if X.ndim == 1:
        # mono to stereo
        X = np.asarray([X, X])

    print('stft of wave source...', end=' ')
    X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
    print('done')

    sp = Separator(model, device, args.batchsize, args.cropsize, args.postprocess)

    if args.tta:
        y_spec, v_spec = sp.separate_tta(X_spec)
    else:
        y_spec, v_spec = sp.separate(X_spec)

    print('validating output directory...', end=' ')
    output_dir = args.output_dir
    if output_dir != "":  # modifies output_dir if theres an arg specified
        output_dir = output_dir.rstrip('/') + '/'
        os.makedirs(output_dir, exist_ok=True)
    print('done')

    print('inverse stft of instruments...', end=' ')
    wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
    print('done')
    sf.write('{}{}_Instruments.wav'.format(output_dir, basename), wave.T, sr)

    print('inverse stft of vocals...', end=' ')
    wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
    print('done')
    sf.write('{}{}_Vocals.wav'.format(output_dir, basename), wave.T, sr)

    if args.output_image:
        image = spec_utils.spectrogram_to_image(y_spec)
        utils.imwrite('{}{}_Instruments.jpg'.format(output_dir, basename), image)

        image = spec_utils.spectrogram_to_image(v_spec)
        utils.imwrite('{}{}_Vocals.jpg'.format(output_dir, basename), image)


if __name__ == '__main__':
    inference(
        r'output\String Theory Explained – What is The True Nature of Reality-_3\en.wav', 
        r'output\String Theory Explained – What is The True Nature of Reality-_3')
