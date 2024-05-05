# Import necessary libraries
import pyaudio
import numpy as np
import time
import subprocess
from collections import deque
import sounddevice as sd
import librosa
import tensorflow as tf
import sys


sys.stdout.reconfigure(encoding='utf-8')

# Load the wake-up word detection model
model_path = 'Models/WakeUpWordModel.keras'
model = tf.keras.models.load_model(model_path)

# Define preprocessing functions
def preprocess_test_voice_wake_up(audio, num_mfcc=25, n_fft=2048, hop_length=512, sr=16000):
    # Resample audio if needed
    if sr != RATE:
        audio = librosa.resample(audio, sr, RATE)
    # Extract MFCC features
    mfccs = extract_mfcc(audio=audio, num_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    features = np.expand_dims(mfccs, axis=0)  # Add batch dimension
    features = np.expand_dims(features, axis=-1)
    return features

def extract_mfcc(audio, num_mfcc=25, n_fft=2048, hop_length=512, target_length=3):
    sr = 16000
    num_frames = int(sr * target_length)

    if len(audio) > num_frames:
        audio = audio[:num_frames]
    else:
        audio = np.pad(audio, (0, num_frames - len(audio)))
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs = librosa.util.normalize(mfccs)
    return mfccs

# Define audio recording parameters
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_DURATION = 4  # Duration of audio to be kept (in seconds)

# Define audio playback function
def play_audio(audio_data, sample_rate=16000):
    print("Playing...")
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()  # Wait for playback to complete
    print("Playback finished.")

# Circular buffer to store audio frames
buffer = deque(maxlen=int(RATE / CHUNK * RECORD_DURATION))

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open stream
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("Recording...")

try:
    while True:
        # Read audio data
        data = stream.read(CHUNK)
        
        # Add audio data to circular buffer
        buffer.append(np.frombuffer(data, dtype=np.float32))
        
        # Check if buffer is filled
        if len(buffer) == buffer.maxlen:
            audio_data = np.concatenate(buffer)
            preprocessed_audio = preprocess_test_voice_wake_up(audio_data)
            prediction = model.predict(preprocessed_audio)

            if prediction > 0.9:
                print("Wake-up word detected!")
                play_audio(audio_data=audio_data)
                break

        
finally:
    # Close the stream and terminate PyAudio
    stream.stop_stream()
    stream.close()
    audio.terminate()
