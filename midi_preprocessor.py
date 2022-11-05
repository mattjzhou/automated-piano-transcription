import os

import pretty_midi
import numpy as np

from audio_preprocessor import load_metadata

# having max 512 events for each spec sequence seems reasonable
SEQ_SIZE = 512


# converts a midi note to two events, note_on and note_off
# each array has format [time, pitch, velocity, note_off]
def note_to_array(note):
    note_on = np.array([note.start, note.pitch, note.velocity, 0])
    note_off = np.array([note.end, note.pitch, note.velocity, 1])
    return note_on, note_off


def midi_to_array(path):
    midi = pretty_midi.PrettyMIDI(path)
    midi_seq = []
    for ins in midi.instruments:
        for note in ins.notes:
            note_on, note_off = note_to_array(note)
            midi_seq.append(note_on)
            midi_seq.append(note_off)
    midi_seq = np.array(midi_seq)
    return midi_seq[midi_seq[:, 0].argsort()]


def midi_to_array_by_sequence(midi_path, times):
    midi = midi_to_array(midi_path)
    split_midi = []
    for seq in times:
        start_time = seq[0]
        end_time = seq[-2]
        seq_midi = midi[midi[:, 0] >= start_time]
        seq_midi = seq_midi[seq_midi[:, 0] <= end_time]
        np.append(seq_midi, np.zeros(4))
        seq_midi = np.pad(seq_midi, ((0, SEQ_SIZE - seq_midi.shape[0] % SEQ_SIZE),(0,0)))
        split_midi.append(seq_midi)
    split_midi = np.array(split_midi)

    return split_midi


def load_time(metadata, i, proj_path, save_path):
    filename = ''
    if metadata['split'][str(i)] == 'train':
        os.chdir(save_path + 'train/times/')
        filename = 'train_time_' + str(i) + '.npy'
    elif metadata['split'][str(i)] == 'test':
        os.chdir(save_path + 'test/times/')
        filename = 'test_time_' + str(i) + '.npy'
    elif metadata['split'][str(i)] == 'validation':
        os.chdir(save_path + 'val/times/')
        filename = 'val_time_' + str(i) + '.npy'
    times = np.load(filename)
    os.chdir(proj_path)
    return times


def save_midi_as_npy(data_path, metadata, proj_path, save_path):
    filename = ''
    for i in range(len(metadata['duration'])):
        times = load_time(metadata, i, proj_path, save_path)
        midi = midi_to_array_by_sequence(data_path + metadata['midi_filename'][str(i)], times)
        if metadata['split'][str(i)] == 'train':
            filename = 'train_midi_' + str(i)
            os.chdir(save_path + 'train/midi/')
        elif metadata['split'][str(i)] == 'test':
            filename = 'train_midi_' + str(i)
            os.chdir(save_path + 'test/midi/')
        elif metadata['split'][str(i)] == 'validation':
            filename = 'train_midi_' + str(i)
            os.chdir(save_path + 'val/midi/')
        np.save(filename, midi)
        os.chdir(proj_path)


if __name__ == '__main__':
    data_path = 'data/maestro-v3.0.0/'
    meta = load_metadata(data_path + 'maestro-v3.0.0.json')
    midi = pretty_midi.PrettyMIDI(data_path + meta['midi_filename']['0'])
    for ins in midi.instruments:
        for note in ins.notes:
            print(note)
    proj_path = "C:/Users/Andrew/Documents/GitHub/Deep-Learning-Project"
    save_path = "D:/dlp/"
    # save_midi_as_npy(data_path, meta, proj_path, save_path)
