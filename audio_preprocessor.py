import json
import os

import librosa
import torch
# torchaudio seems to be faster on my machine; librosa and Google's ddsb exists
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# default params from T3 paper, may need to adjust based on memory requirements
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


# basically copied this, but now wondering why this exists
def _audio_to_frames(samples):
    """Convert audio samples to non-overlapping frames and frame times."""
    frame_size = HOP_WIDTH
    # print(('Padding %d samples to multiple of %d' % (len(torch.flatten(samples)), frame_size)))
    samples = np.pad(torch.flatten(samples), [0, frame_size - len(samples) % frame_size], mode='constant')
    frames = tf.signal.frame(torch.tensor(samples), frame_length=HOP_WIDTH, frame_step=HOP_WIDTH, pad_end=True)
    num_frames = len(samples) // frame_size
    # print('Encoded %d samples to %d frames (%d samples each)' % (len(samples), num_frames, frame_size))
    times = np.arange(num_frames) / frames_per_second
    # returns tensor of shape (num_frames, HOP_WIDTH) and times of length(num_frames)
    # this is needed due to using both pytorch and tensorflow, since only tensorflow has the frame function
    return torch.tensor(frames.numpy()), times


def load_metadata(path):
    try:
        with open(path, 'r') as f:
            metadata = json.load(f)
            return metadata
    except OSError:
        print("Could not open/read file:", path)


def tokenize(path):
    samples = load_wav(path)
    frames, times = _audio_to_frames(samples)
    return frames, times


# copied from torchaudio docs
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


# this is probably less space than the audio, and allows storing of processed data on hard disk rather than memory
# samples are raw sequences separated on frames in shape (frames, times)
# times are sequences of the real times of each frame
# specs are the mel spectrograms in the shape (frames, mel_bins)
def wav_to_save(path, metadata, spectrogram):
    train, test, val = 'train', 'test', 'val'
    samplepath, timepath, specpath = '', '', ''
    for i in range(len(metadata['duration'])):
        samples, times = tokenize(path + metadata['audio_filename'][str(i)])
        # transpose for shape (frames, mel_bins)
        spec = spectrogram(torch.reshape(samples, [-1])).numpy().T
        samples = samples.numpy()
        if metadata['split'][str(i)] == 'train':
            samplepath = train + '_raw_' + str(i)
            timepath = train + '_time_' + str(i)
            specpath = train + '_spec_' + str(i)
            os.chdir("D:/dlp/train/")
        elif metadata['split'][str(i)] == 'test':
            samplepath = test + '_raw_' + str(i)
            timepath = test + '_time_' + str(i)
            specpath = test + '_spec_' + str(i)
            os.chdir("D:/dlp/test/")
        elif metadata['split'][str(i)] == 'validation':
            samplepath = val + '_raw_' + str(i)
            timepath = val + '_time_' + str(i)
            specpath = val + '_spec_' + str(i)
            os.chdir("D:/dlp/val/")
        np.save(samplepath, samples)
        np.save(timepath, times)
        np.save(specpath, spec)
        os.chdir("C:/Users/Andrew/Documents/GitHub/Deep-Learning-Project")


if __name__ == "__main__":
    data_path = 'data/maestro-v3.0.0/'
    meta = load_metadata(data_path + 'maestro-v3.0.0.json')
    # example = wav_to_dict(data_path, meta)
    # samplesm = load_wav(data_path + meta['audio_filename']['0'])
    # print(samplesm)
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=FFT_SIZE,
        hop_length=HOP_WIDTH,
        n_mels=MEL_BINS
    )



