import os
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# configure allowed extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# load pre-trained model
model_path = os.path.abspath('env\\Scripts\\modelENV_skin_final.h5')
model = tf.keras.models.load_model(model_path)

# define function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# define function to predict disease type
def predict_disease_type(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(125, 125))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    categories = ['Atopic Dermatitis', 'Basal Cell Carcinoma', 'Seborrheic Keratoses', 'Eczema', 'Melanocytic Nevi', 'Melanoma', 'Psoriasis and Lichen Planus', 'Benign Keratosis-like Lesions (BKL)', 'Tinea Ringworm Candidiasis', 'Warts Molluscum']
    index = np.argmax(pred)
    prediction = categories[index]
    confidence_score = pred[0][index]
    percentage = round(confidence_score, 0) * 100
    return prediction, prediction, percentage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # check if file was uploaded
        if 'files' not in request.files:
            return render_template('index.html', error='Please select an image.')

        file = request.files['files']

        # check if file was selected
        if file.filename == '':
            return render_template('index.html', error='Please select a new image.')

        # check if file is allowed
        if not allowed_file(file.filename):
            return render_template('index.html', error='Allowed file types are jpg, jpeg, and png.')

        # save file to upload folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # predict tumor type
        tumor_type, result_class, confidence_score = predict_disease_type(file_path)
        return jsonify({
            'sucess': True,
            'filename': filename,
            'result':tumor_type,
            'result_class': result_class,
            'confidence_score':confidence_score
        }), 200

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)