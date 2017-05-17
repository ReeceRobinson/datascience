#!/usr/bin/env python3
"""Show a text-mode spectrogram using live microphone data."""
import argparse
import math
import numpy as np
import shutil
import librosa
import time

class RingBuffer:
    """RingBuffer class"""

    def __init__(self, chunk_length, num_chunks, bytes_per_frame, sample_rate):
        self.chunk_length = chunk_length
        self.ring_length = num_chunks*chunk_length
        self.ring_buffer = np.zeros(self.ring_length, dtype=np.float32)
        # debug
        self.frames = 0
        self.sample_rate = sample_rate
    
    # Test adding to ring buffer
    def add_chunk(self, data):
        d = data.ravel()
        self.ring_buffer = np.roll(self.ring_buffer, len(d))
        self.ring_buffer[:len(d)] = d[::-1]
        self.frames += 1

    def read_window(self, size, offset):
        num_samples = int(size)
        offset_samples = int(offset)
        return self.ring_buffer[offset_samples : (num_samples+offset_samples)]
        
    def print_buffer(self):
        print(self.ring_buffer)

    def print_debug(self):
        print(self.frames)

    def save_ring_buffer(self, sample_rate, filename = "ring_buffer.wav"):
        print("Ring buffer size: %i, sample rate: %s"%(len(self.ring_buffer), sample_rate))
        librosa.output.write_wav(filename, self.ring_buffer[::-1], int(sample_rate))


class ResultRingBuffer:
    def __init__(self, size, item_size):
        self.size = size
        self.ring_buffer = np.zeros((size,item_size))
        self.count = 0
    
    def add_mel(self, mel):
        self.ring_buffer = np.roll(self.ring_buffer, 1, axis=0)
        self.ring_buffer[0] = mel
        self.count += 1

    def get_mels_for_window(self, num_mels):
        return self.ring_buffer[:num_mels][::-1]

    def save_ring_buffer(self, filename="mel_ring_buffer.txt"):
        np.savetxt(filename,self.ring_buffer[::-1])


usage_line = ' press <enter> to quit, +<enter> or -<enter> to change scaling '
frames_log = np.array([])

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


try:
    columns, _ = shutil.get_terminal_size()
except AttributeError:
    columns = 80

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-l', '--list-devices', action='store_true',
                    help='list audio devices and exit')
parser.add_argument('-b', '--block-duration', type=float,
                    metavar='DURATION', default=25,
                    help='block size (default %(default)s milliseconds)')
parser.add_argument('-c', '--columns', type=int, default=columns,
                    help='width of spectrogram')
parser.add_argument('-d', '--device', type=int_or_str,
                    help='input device (numeric ID or substring)')
parser.add_argument('-g', '--gain', type=float, default=10,
                    help='initial gain factor (default %(default)s)')
parser.add_argument('-r', '--range', type=float, nargs=2,
                    metavar=('LOW', 'HIGH'), default=[200, 8000],
                    help='frequency range (default %(default)s Hz)')
args = parser.parse_args()

low, high = args.range
if high <= low:
    parser.error('HIGH must be greater than LOW')

ring = RingBuffer(1200, 500, 1, 48000)

mel_resample_rate = 16000
num_mels = 40
mel_stride = 0.01
mel_width = 0.025
target_window_length = 3.5 # seconds
mel_ring_size = int((target_window_length / mel_stride) + 1)
mel_ring = ResultRingBuffer(mel_ring_size, num_mels)

# Create a nice output gradient using ANSI escape sequences.
# Stolen from https://gist.github.com/maurisvh/df919538bcef391bc89f
colors = 30, 34, 35, 91, 93, 97
chars = ' :%#\t#%:'
gradient = []
for bg, fg in zip(colors, colors[1:]):
    for char in chars:
        if char == '\t':
            bg, fg = fg, bg
        else:
            gradient.append('\x1b[{};{}m{}'.format(fg, bg + 10, char))

try:
    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)

    samplerate = sd.query_devices(args.device, 'input')['default_samplerate']

    delta_f = (high - low) / (args.columns - 1)
    fftsize = math.ceil(samplerate / delta_f)
    low_bin = math.floor(low / delta_f)

    def calc_mel_for_frame(x):
        x_16k = librosa.resample(args.gain * x,samplerate, mel_resample_rate)
        frame = x_16k * np.hamming(len(x_16k))
        frame_magnitude = np.abs(np.fft.rfft(frame, n=fftsize))
        frame_power = ( 1 / fftsize ) * (frame_magnitude ** 2 )
        mel_basis = librosa.filters.mel(mel_resample_rate, fftsize, n_mels=num_mels, fmin=low, fmax=high, norm=1)
        return np.dot(mel_basis, frame_power)

    def calc_mel_for_window(size):
        x = ring.read_window(size)
        x_16k = librosa.resample(x,samplerate, mel_resample_rate)
        S = librosa.feature.melspectrogram(x_16k, sr=mel_resample_rate, n_mels=num_mels, fmin=low, fmax=high, hop_length=int(samplerate/mel_stride))

        # Convert to log scale (dB). We'll use the peak power as reference.
        log_S = librosa.logamplitude(S, ref_power=np.max)
        return log_S

    def print_ascii_mel(mel):
        line = (gradient[int(np.clip(x, 0, 1) * (len(gradient) - 1))] for x in mel[low_bin:low_bin + args.columns])
        print(*line, sep='', end='\x1b[0m\n')

    def callback(indata, frames, time, status):
        global frames_log
        if status: 
            text = ' ' + str(status) + ' '
            print('\x1b[34;40m', text.center(args.columns, '#'),
                  '\x1b[0m', sep='')
        if any(indata):
            
            ring.add_chunk(indata)

            if( ring.frames > 0 and ring.frames%2 == 0):
                # every other frame
                x = ring.read_window(mel_width * samplerate, 2 * mel_stride * samplerate) 
                mel = calc_mel_for_frame(x)
                mel_ring.add_mel(mel)
                print_ascii_mel(mel)

            x = ring.read_window(mel_width * samplerate, mel_stride * samplerate)
            mel = calc_mel_for_frame(x)
            mel_ring.add_mel(mel)
            print_ascii_mel(mel)

            x = ring.read_window(mel_width * samplerate, 0)
            mel = calc_mel_for_frame(x)
            mel_ring.add_mel(mel)
            print_ascii_mel(mel)
                
            # Debug info
            frames_log = np.append(frames_log,frames)
        else:
            print('no input')

    start_time = time.time()
    with sd.InputStream(device=args.device, channels=1, callback=callback,
                        blocksize=int(samplerate * args.block_duration / 1000),
                        samplerate=samplerate):
        while True:
            response = input()
            if response in ('', 'q', 'Q'):
                break
            for ch in response:
                if ch == '+':
                    args.gain *= 2
                elif ch == '-':
                    args.gain /= 2
                else:
                    print('\x1b[31;40m', usage_line.center(args.columns, '#'),
                          '\x1b[0m', sep='')
                    break
except KeyboardInterrupt:
    end_time = time.time()
    mean = np.mean(frames_log)
    std = np.std(frames_log)
    print("%i frames mean: %0.2f, std: %0.2f"%(len(frames_log),mean,std))
    print("mel_ring_size: {}, count: {}".format(mel_ring.size, mel_ring.count))
    ring.save_ring_buffer(samplerate, "audio_ring_buffer.wav")
    mel_ring.save_ring_buffer("mel_ring_buffer.txt")
    print("recording lasted {:0.2f} seconds.".format(end_time - start_time))
    parser.exit('Interrupted by user')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
