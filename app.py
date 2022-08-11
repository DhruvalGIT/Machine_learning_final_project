from keras.models import load_model
import cv2
import numpy as np
from flask import Flask,render_template, request, jsonify

import os
#port = int(os.environ.get('PORT', 5000))
app=Flask(__name__)

model = load_model('cnn_face_mask.h5')


@app.route('/')
def home():
    return render_template('index.html')

if __name__=='__main__':
    app.run()
