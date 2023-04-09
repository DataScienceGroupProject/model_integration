# loading json and creating model
# Import the model_from_json function from Keras
from keras.models import model_from_json
import librosa
import pandas as pd
import numpy as np

def load_model():
    # Open the json file containing the model architecture
    json_file = open('D:\\voice_model_4\\trained_models\\model.json', 'r')
    # Read the contents of the json file
    loaded_model_json = json_file.read()
    # Close the json file
    json_file.close()
    # Recreate the model architecture from the json file
    loaded_model = model_from_json(loaded_model_json)
    # Load the model weights from the h5 file
    loaded_model.load_weights("D:\\voice_model_4\\trained_models\\model.h5")
    # Print message to confirm model was loaded
    print("Loaded model from disk")
    return loaded_model

def extract_audio_features(audio_path,sampling_rate):
    # Load audio from the given path and set the sampling rate
    X, sample_rate = librosa.load(audio_path ,res_type='kaiser_fast',duration=2.5,sr=sampling_rate*2,offset=0.5)
    sample_rate = np.array(sample_rate)

    # Separate harmonic and percussive components of the audio
    y_harmonic, y_percussive = librosa.effects.hpss(X)
    # Extract pitch and magnitudes of the audio
    pitches, magnitudes = librosa.core.pitch.piptrack(y=X, sr=sample_rate)

    # Extract the mean of the Mel-Frequency Cepstral Coefficients (MFCCs) of the audio
    mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=13),axis=1)

    # Extract the mean of the pitches and remove trailing zeroes
    pitches = np.trim_zeros(np.mean(pitches,axis=1))[:20]

    # Extract the mean of the magnitudes and remove trailing zeroes
    magnitudes = np.trim_zeros(np.mean(magnitudes,axis=1))[:20]

    # Extract the mean of the chroma feature of the audio
    chromas = np.mean(librosa.feature.chroma_cqt(y=y_harmonic, sr=sampling_rate),axis=1)
    
    # Return a list of features including the MFCCs, pitches, magnitudes, and chroma feature
    return [mfccs, pitches, magnitudes, chromas]