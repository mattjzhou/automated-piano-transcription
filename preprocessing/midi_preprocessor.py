import os

import pretty_midi
import audio_preprocessor
import numpy as np

SEQ_SIZE = 2048
A0 = 21  # 21 is equivalent to A0, which is the lowest note on a piano, which we want to adjust to be 0
PAD = 0
SOS = 1
EOS = 2
RANGE_NOTE = 88  # 88 keys on a piano, assuming correct midi annotation, no notes outside this range are possible
RANGE_VEL = 128  # 0 velocity is special such that following notes are note_offs
# RANGE_TIME = As high as needed, quantized into 10 ms bins such that 100 = 1 sec

START_IDX = {
    '<pad>': 0,
    '<sos>': 1,
    '<eos>': 2,
    'note': 3,
    'velocity': 3 + RANGE_NOTE,
    'time_set': 3 + RANGE_NOTE + RANGE_VEL
}


class SustainAdapter:
    def __init__(self, time, type):
        self.start = time
        self.type = type


# handle pedaled (sustain/damper/right pedal) notes by extending based the pedal duration
class SustainDownManager:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.managed_notes = []
        self._note_dict = {}  # key: pitch, value: note.start

    def add_managed_note(self, note: pretty_midi.Note):
        self.managed_notes.append(note)

    def transposition_notes(self):
        for note in reversed(self.managed_notes):
            try:
                note.end = self._note_dict[note.pitch]
            except KeyError:
                note.end = max(self.end, note.end)
            self._note_dict[note.pitch] = note.start


# Divided note by note_on, note_off
class SplitNote:
    def __init__(self, note_type, time, value, velocity):
        ## type: note_on, note_off
        self.type = note_type
        self.time = time
        self.value = value
        self.velocity = velocity

    def __repr__(self):
        return '<[SNote] time: {} note_type: {}, value: {}, velocity: {}>' \
            .format(self.time, self.type, self.value, self.velocity)


class Event:
    def __init__(self, event_type, value):
        self.type = event_type
        self.value = value

    def __repr__(self):
        return '<Event type: {}, value: {}>'.format(self.type, self.value)

    def to_int(self):
        return START_IDX[self.type] + self.value

    @staticmethod
    def from_int(int_value):
        info = Event._type_check(int_value)
        return Event(info['type'], info['value'])

    @staticmethod
    def _type_check(int_value):
        range_note = range(3, 3 + RANGE_NOTE)
        range_velocity = range(3 + RANGE_NOTE, 3 + RANGE_NOTE + RANGE_VEL)

        # not sure about this, but I guess keep it in
        valid_value = int_value
        if int_value == PAD:
            return {'type': '<pad>', 'value': valid_value}
        elif int_value == SOS:
            return {'type': '<sos>', 'value': valid_value}
        elif int_value == EOS:
            return {'type': '<eos>', 'value': valid_value}
        elif int_value in range_note:
            valid_value -= 3
            return {'type': 'note', 'value': valid_value}
        elif int_value in range_velocity:
            valid_value -= (3 + RANGE_NOTE)
            return {'type': 'velocity', 'value': valid_value}
        else:  # it's a time event
            valid_value -= (3 + RANGE_NOTE + RANGE_VEL)
            return {'type': 'time_set', 'value': valid_value}


def _divide_note(notes):
    result_array = []
    notes.sort(key=lambda x: x.start)

    for note in notes:
        on = SplitNote('note_on', note.start, note.pitch, note.velocity)
        off = SplitNote('note_off', note.end, note.pitch, 0)
        result_array += [on, off]
    return result_array


def _merge_note(snote_sequence):
    note_on_dict = {}
    result_array = []

    for snote in snote_sequence:
        # print(note_on_dict)
        if snote.type == 'note_on':
            note_on_dict[snote.value] = snote
        elif snote.type == 'note_off':
            try:
                on = note_on_dict[snote.value]
                off = snote
                if off.time - on.time == 0:
                    continue
                result = pretty_midi.Note(on.velocity, snote.value, on.time, off.time)
                result_array.append(result)
            except:
                print('info removed pitch: {}'.format(snote.value))
    return result_array


def _snote_to_events(snote: SplitNote, prev_vel: int):
    result = []
    if prev_vel != snote.velocity:
        result.append(Event(event_type='velocity', value=snote.velocity))
    pitch = snote.value - A0
    result.append(Event(event_type='note', value=pitch))
    return result


def _event_seq_to_snote_seq(event_sequence):
    time = 0
    velocity = 0
    snote_seq = []

    for event in event_sequence:
        thing = event.type + ": " + str(event.value)
        print(thing)
        if event.type == 'time_set':
            # convert from 10 ms to seconds
            time = event.value / 100
        if event.type == 'velocity':
            velocity = event.value
        else:  # it's a note event
            if velocity == 0:
                note_type = 'note_off'
            else:
                note_type = 'note_on'
            pitch = event.value + A0
            snote = SplitNote(note_type, time, pitch, velocity)
            snote_seq.append(snote)
    return snote_seq


def _make_time_set_events(time):
    # multiply seconds time by 100 to get increments of 10 ms
    # round for quantization
    encoded_time = int(np.round(time * 100))
    return [Event(event_type='time_set', value=encoded_time)]


def _control_preprocess(ctrl_changes):
    sustains = []

    manager = None
    for ctrl in ctrl_changes:
        if ctrl.value >= 64 and manager is None:
            # sustain down
            manager = SustainDownManager(start=ctrl.time, end=None)
        elif ctrl.value < 64 and manager is not None:
            # sustain up
            manager.end = ctrl.time
            sustains.append(manager)
            manager = None
        elif ctrl.value < 64 and len(sustains) > 0:
            sustains[-1].end = ctrl.time
    return sustains


def _note_preprocess(sustains, notes):
    note_stream = []

    if sustains:  # if the midi file has sustain controls
        for sustain in sustains:
            # add notes within the sustain
            for note_idx, note in enumerate(notes):
                # if note is before sustain, add it into the note stream
                if note.start < sustain.start:
                    note_stream.append(note)
                # if note is after sustain, this sustain is over and sustained note ends are adjusted
                # we then shift notes array accordingly
                elif note.start > sustain.end:
                    notes = notes[note_idx:]
                    sustain.transposition_notes()
                    break
                else:
                    sustain.add_managed_note(note)
        # add sustained notes
        for sustain in sustains:
            note_stream += sustain.managed_notes

    else:  # else, just push everything into note stream
        for note_idx, note in enumerate(notes):
            note_stream.append(note)

    note_stream.sort(key=lambda x: x.start)
    return note_stream


# encode a section of midi based on a given time array
def encode_midi_with_time(file_path, time):
    notes = []  # array of pretty_midi Notes
    mid = pretty_midi.PrettyMIDI(midi_file=file_path)
    # give events twice the sequence length as input
    seq_len = time.shape[0] * 2

    # here just one, but might as well keep it
    for inst in mid.instruments:
        inst_notes = inst.notes
        # pretty sure 64 (sustain pedal) is the only control sequence in the MAESTRO dataset
        # also if we're just transcribing notes, no need for any other control sequence anyway
        # see https://www.midi.org/specifications-old/item/table-3-control-change-messages-data-bytes-2 for full list
        ctrls = _control_preprocess([ctrl for ctrl in inst.control_changes if ctrl.number == 64])
        notes += _note_preprocess(ctrls, inst_notes)

    # convert notes into on and offs
    dnotes = _divide_note(notes)
    dnotes.sort(key=lambda x: x.time)

    note_seq = []
    start_time = time[0]
    end_time = np.max(time)
    for j in dnotes:
        # probably should inclusive like this
        if start_time <= j.time <= end_time:
            # shift all time relative to the start time of the sequence
            # we want each sequences to start at time 0
            j.time -= start_time
            note_seq.append(j)
        elif j.time > end_time:
            break

    cur_vel = 0
    events = [Event(event_type='<sos>', value=0)]
    for snote in note_seq:
        events += _make_time_set_events(time=snote.time)
        events += _snote_to_events(snote=snote, prev_vel=cur_vel)

        cur_vel = snote.velocity

    encoded_events = [e.to_int() for e in events]
    encoded_events.append(EOS)
    # make sure we don't have any issues here
    if len(encoded_events) > seq_len:
        print(len(encoded_events))
        print(file_path)
    encoded_events = np.pad(np.array(encoded_events), (0, seq_len - len(encoded_events) % seq_len),
                            constant_values=(0, PAD))
    return encoded_events


# testing if we are encoding and decoding correctly for an entire midi file
def encode_midi(file_path):
    notes = []  # array of pretty_midi Notes
    mid = pretty_midi.PrettyMIDI(midi_file=file_path)

    # here just one, but might as well keep it
    for inst in mid.instruments:
        inst_notes = inst.notes
        # pretty sure 64 (sustain pedal) is the only control sequence in the MAESTRO dataset
        # also if we're just transcribing notes, no need for any other control sequence anyway
        # see https://www.midi.org/specifications-old/item/table-3-control-change-messages-data-bytes-2 for full list
        ctrls = _control_preprocess([ctrl for ctrl in inst.control_changes if ctrl.number == 64])
        notes += _note_preprocess(ctrls, inst_notes)

    # convert notes into on and offs
    dnotes = _divide_note(notes)
    dnotes.sort(key=lambda x: x.time)

    cur_vel = 0
    events = [Event(event_type='<sos>', value=0)]
    for snote in dnotes:
        events += _make_time_set_events(time=snote.time)
        events += _snote_to_events(snote=snote, prev_vel=cur_vel)
        cur_vel = snote.velocity

    encoded_events = [e.to_int() for e in events]
    encoded_events.append(EOS)
    # make sure we don't have any issues here
    return encoded_events


def decode_midi(idx_array, file_path=None):
    event_sequence = []
    for idx in idx_array:
        if idx != PAD and idx != SOS and idx != EOS:
            # let's only decode non-padding, sos, and eos
            event_sequence.append(Event.from_int(idx))
    # print(event_sequence)
    snote_seq = _event_seq_to_snote_seq(event_sequence)
    note_seq = _merge_note(snote_seq)
    note_seq.sort(key=lambda x: x.start)
    print(note_seq)
    mid = pretty_midi.PrettyMIDI()
    # if want to change instument, see https://www.midi.org/specifications/item/gm-level-1-sound-set
    instument = pretty_midi.Instrument(1, False, "Developed By Yang-Kichang")
    instument.notes = note_seq

    mid.instruments.append(instument)
    if file_path is not None:
        mid.write(file_path)
    return mid


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


def save_encoded(data_path, metadata, proj_path, save_path):
    filename = ''
    for i in range(len(metadata['duration'])):
        times = load_time(metadata, i, proj_path, save_path)
        midi = encode_midi_with_time(data_path + metadata['midi_filename'][str(i)], times)
        if metadata['split'][str(i)] == 'train':
            filename = 'train_midi_' + str(i)
            os.chdir(save_path + 'train/midi/')
        elif metadata['split'][str(i)] == 'test':
            filename = 'test_midi_' + str(i)
            os.chdir(save_path + 'test/midi/')
        elif metadata['split'][str(i)] == 'validation':
            filename = 'val_midi_' + str(i)
            os.chdir(save_path + 'val/midi/')
        np.save(filename, midi)
        os.chdir(proj_path)


if __name__ == '__main__':
    data_path = '../data/maestro-v3.0.0/'
    meta = audio_preprocessor.load_metadata(data_path + 'maestro-v3.0.0.json')

    proj_path = "/"
    save_path = "D:/dlp/"
    # testing for correct encoding/decoding of midi data
    print(meta['midi_filename']['158'])

    encoded = encode_midi(data_path + meta['midi_filename']['158'])
    print(encoded)
    decode_midi(encoded, '../test.mid')

    # just some testing below
    # tbh, not sure why they don't match, but the actually listening to the midis they seem to be the same
    # our_midi = pretty_midi.PrettyMIDI('test.mid')
    # labeled_midi = pretty_midi.PrettyMIDI(data_path + meta['midi_filename']['158'])

    # for ins in labeled_midi.instruments:
    #     for note in ins.notes:
    #         note.start = np.round(note.start, decimals=2)
    #         note.end = np.round(note.end, decimals=2)
    # our_notes = our_midi.instruments[0].notes
    # labeled_notes = labeled_midi.instruments[0].notes
    # for i in range(len(our_notes)):
    #     our_note = our_notes[i]
    #     labeled_note = labeled_notes[i]
    #     if our_note.start != labeled_note.start:
    #         print("start off")
    #         print(our_note.start)
    #         print(labeled_note.start)
    #     if our_note.end != labeled_note.end:
    #         print("end off")
    #         print(our_note.end)
    #         print(labeled_note.end)
    #     if our_note.velocity != labeled_note.velocity:
    #         print("velocity off")
    #         print(our_note.velocity)
    #         print(labeled_note.velocity)
    #     if our_note.pitch != labeled_note.pitch:
    #         print("pitch off")
    #         print(our_note.pitch)
    #         print(labeled_note.pitch)
