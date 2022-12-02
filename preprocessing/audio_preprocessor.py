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

# default params from MT3 paper, may need to adjust based on memory requirements
SAMPLE_RATE = 16000
HOP_WIDTH = 128
MEL_BINS = 512
FFT_SIZE = 2048
SEQ_SIZE = 511  # + 1 EOS
frames_per_second = SAMPLE_RATE / HOP_WIDTH
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPECTROGRAM = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=FFT_SIZE,
    hop_length=HOP_WIDTH,
    n_mels=MEL_BINS
)


def load_wav(wav):
    # load takes path and returns tensor representation and sample rate
    # tensor is default shape (channels, time)
    samples, _ = torchaudio.load(wav)
    # downsample
    samples = torchaudio.functional.resample(samples, _, SAMPLE_RATE)
    # data is stereo so mean to get mono
    return torch.mean(samples, dim=0)


# basically copied this, but now wondering why this exists
def _audio_to_frames(samples):
    """Convert audio samples to non-overlapping frames and frame times."""
    frame_size = HOP_WIDTH
    # print(('Padding %d samples to multiple of %d' % (len(torch.flatten(samples)), frame_size)))
    samples = np.pad(samples, [0, frame_size - len(samples) % frame_size], mode='constant')
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


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)


# this is probably less space than the audio, and allows storing of processed data on hard disk rather than memory
# samples are raw sequences separated on frames in shape (frames, times)
# times are sequences of the real times of each frame
# specs are the mel spectrograms in the shape (frames, mel_bins)
def wav_to_save(path, metadata, spectrogram):
    train, test, val = 'train', 'test', 'val'
    raw, time, spect = 'raw/', 'times/', 'spec/'
    samplepath, timepath, specpath = '', '', ''
    for i in range(len(metadata['duration'])):
        samples, times = tokenize(path + metadata['audio_filename'][str(i)])
        times = split_time(times)
        # transpose for shape (frames, mel_bins)
        spec = spectrogram(torch.flatten(samples))
        spec = split_spec(spec)
        # samples = samples.numpy()
        if metadata['split'][str(i)] == 'train':
            # samplepath = raw + train + '_raw_' + str(i)
            timepath = time + train + '_time_' + str(i)
            specpath = spect + train + '_spec_' + str(i)
            os.chdir("D:/dlp/train/")
        elif metadata['split'][str(i)] == 'test':
            # samplepath = raw + test + '_raw_' + str(i)
            timepath = time + test + '_time_' + str(i)
            specpath = spect + test + '_spec_' + str(i)
            os.chdir("D:/dlp/test/")
        elif metadata['split'][str(i)] == 'validation':
            # samplepath = raw + val + '_raw_' + str(i)
            timepath = time + val + '_time_' + str(i)
            specpath = spect + val + '_spec_' + str(i)
            os.chdir("D:/dlp/val/")
        # np.save(samplepath, samples)
        print(times.shape)
        print(spec.shape)
        np.save(timepath, times)
        np.save(specpath, spec)
        os.chdir("/")


# splitting everything into sequences is probably most convenient
def split_time(times, seq_len):
    frame_size = len(times)
    pad_times = np.pad(times, (0, seq_len - frame_size % seq_len))
    pad_times = pad_times.reshape((-1, seq_len))
    pad_times = np.pad(pad_times, ((0, 0), (0, 1)))
    return pad_times


def split_spec(spec, seq_len):
    # assumes raw spec, not transposed
    frame_size = spec.shape[1]
    # pad such that we can reshape with groups of spec frames SEQ_SIZE long
    pad_spec = np.pad(spec.numpy().T, ((0, seq_len - frame_size % seq_len), (0, 0)))
    # reshape to split
    pad_spec = pad_spec.reshape((-1, seq_len, MEL_BINS))
    # add EOS as frame of all zeros
    pad_spec = np.pad(pad_spec, ((0, 0), (0, 1), (0, 0)))
    return pad_spec


if __name__ == "__main__":
    data_path = '../data/maestro-v3.0.0/'
    meta = load_metadata(data_path + 'maestro-v3.0.0.json')
    # example = wav_to_dict(data_path, meta)
    print(meta['canonical_title']['158'])
    samplesm = load_wav(data_path + meta['audio_filename']['158'])
    # print(samplesm)
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=FFT_SIZE,
        hop_length=HOP_WIDTH,
        n_mels=MEL_BINS
    )
    plot_waveform(samplesm, SAMPLE_RATE)
    samplesm = samplesm.squeeze()
    spec = mel_spectrogram(samplesm)
    plot_spectrogram(spec)

    # wav_to_save(data_path, meta, mel_spectrogram)
