import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import librosa
import sys
import sounddevice as sd
import queue
import threading
import time

print(tf.__version__)
sys.stdout.reconfigure(encoding='utf-8')

def preprocess_test_voice_wake_up(audio, num_mfcc=25, n_fft=2048, hop_length=512):
    features = extract_mfcc(audio=audio, num_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    features = np.expand_dims(features, axis=0)  # Add batch dimension
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

def record_audio(queue, sample_rate=16000):
    print("Recording thread started.")
    while True:
        audio_data = sd.rec(int(3 * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()  # Wait for recording to complete
        queue.put(audio_data.flatten())
    print("Recording thread stopped.")

def play_audio(audio_data, sample_rate=16000):
    print("Playing...")
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()  # Wait for playback to complete
    print("Playback finished.")

def listen_wake_up_word(model, max_queue=10, sample_rate=16000):
    q = queue.Queue(max_queue)
    record_thread = threading.Thread(target=record_audio, args=(q, sample_rate))
    record_thread.start()

    print("Listening for wake-up word...")
    start_time = time.time()
    while time.time() - start_time < 10:
        if not q.empty():
            recorded_audio = q.get()
            # Check if recorded_audio is not empty and does not contain non-finite values
            if len(recorded_audio) > 0 and np.all(np.isfinite(recorded_audio)):
                preprocessed_audio = preprocess_test_voice_wake_up(recorded_audio)
                prediction = model.predict(preprocessed_audio)

                if prediction > 0.9:
                    print("Wake-up word detected!")
                    play_audio(recorded_audio)
                    return True;
                else:
                    print("No wake-up word detected.")
            else:
                print("Invalid audio data received.")
    print("10sec fini")
    return recorded_audio

# Load model
model_path = 'Models/WakeUpWordModel.keras'
model = tf.keras.models.load_model(model_path)
model.compile(optimizer='adamax',loss='binary_crossentropy')
# Start listening for wake-up word
recorded_audio = listen_wake_up_word(model)
play_audio(recorded_audio)