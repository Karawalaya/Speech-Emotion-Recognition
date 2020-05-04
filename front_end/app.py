import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import os
import pickle
from tensorflow.keras.models import load_model
from werkzeug.utils import redirect
import back_end.predict as p
# from werkzeug.utils import secure_filename

app = Flask(__name__)

uploads_dir = 'uploads'

@app.route('/')
def home():
    return render_template('/upload.html')

@app.route('/uploader', methods = ['POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(os.path.join(uploads_dir, f.filename))
      return redirect(url_for('predict'))

@app.route('/predict',methods=['GET'])
def predict():
    p_path = os.path.join('../back_end/pickles', 'convolutional.p')
    with open(p_path, 'rb') as handle:
        modelconfig = pickle.load(handle)

    print(modelconfig.model_path)
    loaded_model = load_model('../back_end/models/convolutional.model')
    print(loaded_model)
    classes = ['male_negative', 'male_positive']
    p.predict(modelconfig, loaded_model, classes)
    prediction = p.predict(modelconfig, loaded_model, classes)

    output = prediction

    return render_template('upload.html', prediction_text='Emotion should be {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)