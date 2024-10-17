import numpy as np
import torch
from torch.utils.data import Dataset
from audio_preprocessor import load_wav, _audio_to_frames, split_time, split_spec, plot_spectrogram, plot_waveform
from midi_preprocessor import encode_midi_with_time


class MaestroDataset(Dataset):
    def __init__(self, split_idx, metadata, root_dir, seq_len, spectrogram):
        """
        Args:
            split_idx (int array): Indices of the desired split
            metadata (dict): The metadata file with annotations.
            root_dir (string): Directory with the wav and midi files.
            seq_len (int): Hyperparameter, the desired input/output sequence length
            spectrogram (torchaudio spectrogram): The spectrogram config to use
        """
        self.split_idx = split_idx
        # currently using json, can probably work with the csv too
        self.metadata = metadata
        self.root_dir = root_dir
        self.seq_len = seq_len - 1
        self.spectrogram = spectrogram

    def __len__(self):
        return len(self.split_idx)

    def __getitem__(self, idx):
        idx = self.split_idx[idx]
        wav_path = self.root_dir + self.metadata['audio_filename'][str(idx)]
        midi_path = self.root_dir + self.metadata['midi_filename'][str(idx)]

        samples = load_wav(wav_path)
        samples, times = _audio_to_frames(samples)
        plot_waveform(samples)
        times = split_time(times, self.seq_len)
        spec = self.spectrogram(torch.flatten(samples))
        print(spec.shape)
        plot_spectrogram(spec)
        spec = split_spec(spec, self.seq_len)

        # get item probably expects one sequence, idk how better to do it but let's just return one random sample
        seq_idx = np.random.randint(0, times.shape[0])
        spec = spec[seq_idx]

        # save some computation here for midi processing
        time = times[seq_idx]
        midi = encode_midi_with_time(midi_path, time)

        # we don't want attention attending to padding so we mask those indices
        src_key_padding_mask = torch.BoolTensor(np.where(time == 0, True, False))
        tgt_key_padding_mask = torch.BoolTensor(np.where(midi == 0, True, False))

        sample = {'spectrogram': spec,
                  'midi': torch.LongTensor(midi),
                  'src_pad_mask': src_key_padding_mask,
                  'tgt_pad_mask': tgt_key_padding_mask}

        return sample
