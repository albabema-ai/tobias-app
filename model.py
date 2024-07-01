import tensorflow as tf
import numpy as np
import librosa
import os

def compare_sounds(file_path):
    model = tf.saved_model.load('./')
    classes = ["Abnormal", "Normal"]
    waveform, sr = librosa.load(file_path, sr=16000)
    if waveform.shape[0] % 16000 != 0:
        waveform = np.concatenate([waveform, np.zeros(16000 - waveform.shape[0] % 16000)])
    inp = tf.constant(np.array([waveform]), dtype='float32')
    class_scores = model(inp)[0].numpy()
    
    file_name = os.path.basename(file_path)  # Extract only the file name
    
    result_str = (
        f"File: {file_name}\n"
        f"Abnormal: {class_scores[0]}\n"
        f"Normal: {class_scores[1]}\n"
        f"The phonocardiogram is {classes[class_scores.argmax()]}"
    )
    
    return result_str
