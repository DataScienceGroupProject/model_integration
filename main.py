import feature_extraction
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

#declared an empty variable for reassignment
response = ''

#creating the instance of our flask application
app = Flask(__name__)

@app.route('/api',methods=['GET'])

def getEmotion():
    emotion = {}
    wav_file_name = str(request.args['query'])
    demo_audio_path = f"E:\\audio\\{wav_file_name}"
    loaded_model = feature_extraction.load_model()

    # Extract the audio features (MFCC, pitch, magnitude, and chroma)
    demo_mfcc, demo_pitch, demo_mag, demo_chrom = feature_extraction.extract_audio_features(demo_audio_path,20000)

    # Convert the audio features to Pandas Series
    mfcc = pd.Series(demo_mfcc)
    pit = pd.Series(demo_pitch)
    mag = pd.Series(demo_mag)
    C = pd.Series(demo_chrom)
    # Concatenate the audio features into a single dataframe
    demo_audio_features = pd.concat([mfcc,pit,mag,C],ignore_index=True)

    # Add an additional dimension to the demo_audio_features array along the first axis (axis=0)
    demo_audio_features= np.expand_dims(demo_audio_features, axis=0)
    # Add another dimension to the demo_audio_features array along the second axis (axis=2)
    demo_audio_features= np.expand_dims(demo_audio_features, axis=2)

    # Get the shape of the demo_audio_features array
    demo_audio_features.shape

    # Use the loaded model to make predictions on the demo_audio_features
    live_predictions = loaded_model.predict(demo_audio_features, 
                         batch_size=32,
                        # Set verbose to 1 to print progress information
                         verbose=1)
    
    live_predictions

    emotions=["anger","disgust","fear","happy","neutral", "sad", "surprise"]
    # Get the index of the emotion with the highest probability
    index = live_predictions.argmax(axis=1).item()
    index

    answer = emotions[index]
    emotion['output'] = answer
    return emotion

if __name__ == "__main__":
    app.run()