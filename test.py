import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import librosa
import sys
import sounddevice as sd
print(tf.__version__)
sys.stdout.reconfigure(encoding='utf-8')
def preprocess_test_voice_wake_up(audio, num_mfcc=25, n_fft=2048, hop_length=512):
    features = extract_mfcc(audio=audio, num_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    features = np.expand_dims(features, axis=-1)
    print(features.shape)
    return features

def extract_mfcc(audio, num_mfcc=25, n_fft=2048, hop_length=512,target_length = 3):
    sr = 16000
    numFrames = int(sr*target_length)

    if len(audio) > numFrames:
        audio = audio[:numFrames]
    else:
        audio = np.pad(audio, (0, numFrames - len(audio)))
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs = librosa.util.normalize(mfccs)
    return mfccs

model_path = 'Models/WakeUpWordModel.keras'

# Load the model with custom objects
model = tf.keras.models.load_model(model_path)
model.compile(optimizer='adamax',
              loss='binary_crossentropy',
              metrics=['BinaryAccuracy'])

def record_audio(duration=3, sample_rate=16000):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait for recording to complete
    print("Recording finished.")
    print(audio_data.shape)
    return audio_data.flatten()  # Flatten to 1D array


def play_audio(audio_data, sample_rate=16000):
    print("Playing...")
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()  # Wait for playback to complete
    print("Playback finished.")


def ListenWakeUpWord(maxqueue = 3,sample_rate = 16000):
    sd.Stream

recorded_audio = record_audio()

preprocessed_audio = preprocess_test_voice_wake_up(recorded_audio)

play_audio(recorded_audio)




prediction = model.predict(preprocessed_audio)
print(prediction)
if prediction > 0.9:
    print("Correct Keyword")
else:
    print("incorrect Keyword")
