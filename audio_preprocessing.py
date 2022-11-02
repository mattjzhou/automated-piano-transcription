import json

import librosa
import torch
import torch.nn.functional as F
# torchaudio arbitrarily chosen; librosa and Google's ddsb exists
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# default params from T3 paper
SAMPLE_RATE = 16000
HOP_WIDTH = 128
MEL_BINS = 512
FFT_SIZE = 2048
frames_per_second = SAMPLE_RATE / HOP_WIDTH
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_wav(wav):
    # load takes path and returns tensor representation and sample rate
    # tensor is default shape (channels, time)
    samples, _ = torchaudio.load(wav)
    # downsample
    samples = torchaudio.functional.resample(samples, _, SAMPLE_RATE)
    # data is stereo so mean to get mono
    return torch.mean(samples, dim=0, keepdim=True)


def _audio_to_frames(samples):
    """Convert audio samples to non-overlapping frames and frame times."""
    frame_size = HOP_WIDTH

    print(('Padding %d samples to multiple of %d' % (len(torch.flatten(samples)), frame_size)))
    samples = np.pad(torch.flatten(samples), [0, frame_size - len(samples) % frame_size], mode='constant')
    frames = tf.signal.frame(torch.tensor(samples), frame_length=HOP_WIDTH, frame_step=HOP_WIDTH, pad_end=True)
    num_frames = len(samples) // frame_size
    print('Encoded %d samples to %d frames (%d samples each)' % (len(samples), num_frames, frame_size))
    times = np.arange(num_frames) / frames_per_second
    return frames, times


def load_metadata(path):
    try:
        with open(path, 'r') as f:
            metadata = json.load(f)
            return metadata
    except OSError:
        print("Could not open/read file:", path)


# will probably delete and do in tokenize
def wav_to_dict(path, metadata):
    train = test = val = dict()
    for i in range(len(metadata['duration'])):
        if metadata['split'] == 'train':
            train[str(i)] = load_wav(path + metadata['audio_filename'][str(i)]).tolist()
        elif metadata['split'] == 'test':
            test[str(i)] = load_wav(path + metadata['audio_filename'][str(i)]).tolist()
        elif metadata['split'] == 'validation':
            val[str(i)] = load_wav(path + metadata['audio_filename'][str(i)]).tolist()
    return train, test, val


# TODO
def tokenize(path):
    pass


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


if __name__ == "__main__":
    data_path = 'data/maestro-v3.0.0/'
    meta = load_metadata(data_path + 'maestro-v3.0.0.json')
    # example = wav_to_dict(data_path, meta)

    samplesm = load_wav(data_path + meta['audio_filename']['0'])
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=FFT_SIZE,
        hop_length=HOP_WIDTH,
    )

    melspec = mel_spectrogram(samplesm)
    print(torchaudio.info(data_path + meta['audio_filename']['0']))
    print(samplesm.shape)
    framesm, timesm = _audio_to_frames(samplesm)
    print(framesm.shape)
    print(timesm.shape)
    print(framesm[0])
    print(melspec.shape)
    plot_spectrogram(
        melspec[0], title="MelSpectrogram - torchaudio", ylabel='mel freq')
    # dict_to_json(data_path, example)
