from PIL import Image
import os
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow import keras
from werkzeug.utils import secure_filename


app = Flask(__name__)
model= keras.models.load_model("animal.h5")


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods = ["POST", "GET"])
def predict():
    if request.method == "POST":
        f = request.files['img']
        filename = secure_filename(f.filename)
        f.save(os.path.join('/', filename))
        print(os.path.join('/', filename))
        img=image.load_img(os.path.join('/', filename),target_size=(64,64))
        y=image.img_to_array(img)
        y=np.expand_dims(y,axis=0)
        pred=np.argmax(model.predict(y))
        animals=['bears','crows' 'elephants', 'racoons', 'rats']
        print("animal:", animals[pred])
        return {"animal":animals[pred] }  
        
if __name__ == "__main__":
    app.run(debug = True)