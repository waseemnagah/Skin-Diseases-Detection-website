import os
from flask import Flask, request, render_template, redirect, abort, jsonify, flash, url_for
import webbrowser
from flask_cors import CORS
from flask_mail import Mail, Message
from sqlalchemy import or_
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from flask import send_from_directory
from keras.preprocessing import image
import matplotlib.pyplot as plt

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# intial program

app = Flask(__name__)
"""
@app.route("/")
def home():
    return render_template("test2.html")
"""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["POST", "GET"])
def home():
    # check if the post request has the file part
    if request.method == 'POST':
        if 'files' not in request.files:
            flash('No file part')
        file = request.files['files']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        categories = ['Bengin cases', 'Malignant cases', 'Normal cases']
        #categories = ['Astrocitoma', 'Carcinoma', 'Ependimoma', 'Ganglioglioma', 'Germinoma', 'Glioblastoma', 'Granuloma', 'Meduloblastoma', 'Meningioma', 'Neurocitoma', 'Normal', 'Oligodendroglioma', 'Papiloma', 'Schwannoma', 'Tuberculoma']
        model =  tf.keras.models.load_model('F:\\College\\Level 4\\Semester 2\\Pattern Recognition\\Project\\env\\final_model.h5')
        
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img = image.load_img(path, target_size=(256, 256))
        x=image.img_to_array(img)
        x /= 255
        x=np.expand_dims(x, axis=0)
        images = np.vstack([x])
        
        pred = model.predict(images, batch_size=10)
        output = categories[np.argmax(pred)]
        #percentage = round(classes[0][0] * 100, 2)
        
        prediction = output
        #percentage = 100 - percentage

        return render_template("test2.html" , pre=prediction)

# to run app
if __name__ == "__main__":
    # automatically open web browser
    webbrowser.open_new('http://127.0.0.1:5000/')
    app.run()
