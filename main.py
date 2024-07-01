from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import librosa
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your specific domains if necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_location = f"uploaded_files/{file.filename}"
    os.makedirs(os.path.dirname(file_location), exist_ok=True)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Call the AI model script to process the file
    result = compare_sounds(file_location)
    return JSONResponse(content={"result": result})
