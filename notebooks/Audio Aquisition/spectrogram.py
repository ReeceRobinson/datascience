#!/usr/bin/env python3
"""Show a text-mode spectrogram using live microphone data."""
import argparse
import math
import numpy as np
import shutil
import librosa
import time
import sounddevice as sd
import sys
import tensorflow as tf

class RingBuffer:
    """
    This class provides a configurable length ndarray based byte ring buffer for raw audio data.
    """

    def __init__(self, chunk_length, num_chunks, sample_rate):
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
    """
    This class provides a configurable length ndarray based ring buffer for mel spectrogram calculation results.
    """
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

    def get_mels(self):
        return self.ring_buffer[:][::-1]

    def save_ring_buffer(self, filename="mel_ring_buffer.txt"):
        np.savetxt(filename,self.ring_buffer[::-1])

class InferenceEngine:
    """
    This class provides predictions based on the pre-trained tensorflow model.
    """

    def __init__(self):
        # pre-trained RNN model for keyword spotting
        self.model_file = r'kws_model-85-20170906102155.ckpt.meta'
        self.params_file = r'kws_model-85-20170906102155.ckpt'

        # Model Parameters
        self.n_steps = 36
        self.n_inputs = 40
        self.mean = -25.664221
        self.std = 10.932781

        self.saver = tf.train.import_meta_graph(self.model_file)
        self.sess = tf.Session()
        self.saver.restore(self.sess, self.params_file)
        self.graph = tf.get_default_graph()

        print('InferenceEngine initalised.')

    def predict(self):
        def normalize_with_paras(test, mean, std):
            test = (test - mean) / std
            return test

        with self.sess.as_default() as sess:
            assert tf.get_default_session() is sess

            X_raw = mel_ring.get_mels().reshape(1,self.n_steps,self.n_inputs)

            predicted = self.graph.get_collection("predicted")[0]
            X = self.graph.get_collection("X")[0]
            X_feed = normalize_with_paras(X_raw, self.mean, self.std).astype('float32')
            pred = predicted.eval(feed_dict={X:X_feed})

            # print('prediction:',pred)
            return pred[0]
#
# Main program logic
#
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
                    metavar='DURATION', default=100,
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

#
# Capture parameters
#
try:
    rec_sample_rate = sd.query_devices(args.device, 'input')['default_samplerate']
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))

target_window_length = 3.5 # seconds
mel_resample_rate = 16000
num_mels = 40
mel_stride = 0.1
mel_width = 0.1
mel_ring_size = int((target_window_length / mel_stride) + 1)

mel_ring = ResultRingBuffer(mel_ring_size, num_mels)

audio_ring_size = int(rec_sample_rate*mel_width)
audio_chunk_count = int(1/mel_width)*10 # ten seconds

audio_ring = RingBuffer(audio_ring_size, audio_chunk_count, rec_sample_rate)

inferenceEngine = InferenceEngine()
#
# Processing loop attributes
#
stride_frac_residual = 0
mel_stride_bytes = mel_stride * rec_sample_rate

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

    pred_counter = 0
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)

    delta_f = (high - low) / (args.columns - 1)
    fftsize = math.ceil(rec_sample_rate / delta_f)
    low_bin = math.floor(low / delta_f)

    def run_inferencing():
        return inferenceEngine.predict()

    def calc_mel_for_frame(x):
        x_16k = librosa.resample(args.gain * x,rec_sample_rate, mel_resample_rate)
        frame = x_16k * np.hamming(len(x_16k))
        frame_magnitude = np.abs(np.fft.rfft(frame, n=fftsize))
        frame_power = ( 1 / fftsize ) * (frame_magnitude ** 2 )
        mel_basis = librosa.filters.mel(mel_resample_rate, fftsize, n_mels=num_mels, fmin=low, fmax=high, norm=1)
        return np.dot(mel_basis, frame_power)

    def calc_mel_for_window(size):
        x = audio_ring.read_window(size)
        x_16k = librosa.resample(x,rec_sample_rate, mel_resample_rate)
        S = librosa.feature.melspectrogram(x_16k, sr=mel_resample_rate, n_mels=num_mels, fmin=low, fmax=high, hop_length=int(rec_sample_rate/mel_stride))

        # Convert to log scale (dB). We'll use the peak power as reference.
        log_S = librosa.logamplitude(S, ref_power=np.max)
        return log_S

    def print_ascii_mel(mel, prediction):
        global pred_counter

        line = (gradient[int(np.clip(x, 0, 1) * (len(gradient) - 1))] for x in mel[low_bin:low_bin + args.columns])
        print(*line, sep='', end='\x1b[0m')
        if prediction and pred_counter >= 4:
            print('\033[1m\033[95m'+str(prediction), end=' ')
        else:
            print(prediction, end = ' ')
        print(pred_counter)



    def callback(indata, frames, time, status):
        global frames_log
        global stride_frac_residual
        global pred_counter
        prediction = 0

        def inc_counter():
            global pred_counter
            pred_counter = pred_counter + 1

        def reset_counter():
            global pred_counter
            pred_counter =  0

        if status: 
            text = ' ' + str(status) + ' '
            print('\x1b[34;40m', text.center(args.columns, '#'),
                  '\x1b[0m', sep='')
        if any(indata):
            
            audio_ring.add_chunk(indata)

            in_len = len(indata)
            stride_fractional, stride_whole = math.modf(in_len/mel_stride_bytes)
            stride_frac_residual += stride_fractional

            for i in range(0,int(stride_whole)):
                x = audio_ring.read_window(mel_width * rec_sample_rate, i * mel_stride_bytes )
                mel = calc_mel_for_frame(x)
                mel_ring.add_mel(mel)
                prediction = run_inferencing()
                print_ascii_mel(mel, prediction)

            if prediction:
                inc_counter()
            else: # reset back to zero
                reset_counter()
                
            if stride_frac_residual >= 1.0:
                x = audio_ring.read_window(mel_width * rec_sample_rate, int(stride_whole) * mel_stride_bytes)
                mel = calc_mel_for_frame(x)
                mel_ring.add_mel(mel)
                run_inferencing()
                print_ascii_mel(mel, prediction)
                stride_frac_residual -= 1.0

                if prediction:
                    inc_counter()
                else: # reset back to zero
                    reset_counter()

            # Debug info
            frames_log = np.append(frames_log,frames)
        else:
            print('no input')

    start_time = time.time()
    with sd.InputStream(device=args.device, channels=1, callback=callback,
                        blocksize=int(rec_sample_rate * args.block_duration / 1000),
                        samplerate=rec_sample_rate):
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
    audio_ring.save_ring_buffer(rec_sample_rate, "audio_ring_buffer.wav")
    mel_ring.save_ring_buffer("mel_ring_buffer.txt")
    print("recording lasted {:0.2f} seconds.".format(end_time - start_time))
    parser.exit('Interrupted by user')
except Exception as e:

    parser.exit(type(e).__name__ + ': ' + str(e))
