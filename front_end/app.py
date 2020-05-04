import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import os
import pickle
from tensorflow.keras.models import load_model
from werkzeug.utils import redirect
import back_end.predict as p
# from werkzeug.utils import secure_filename

app = Flask(__name__)

uploads_dir = 'test'

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
    '''
    For rendering results on HTML GUI
    '''

    p_path = os.path.join('pickles', 'convolutional.p')
    with open(p_path, 'rb') as handle:
        modelconfig = pickle.load(handle)

    print(modelconfig.model_path)
    loaded_model = load_model('models/convolutional.model')
    print(loaded_model)
    classes = ['male_negative', 'male_positive']
    p.predict(modelconfig, loaded_model, classes)
    prediction = p.predict(modelconfig, loaded_model, classes)

    output = prediction

    return render_template('upload.html', prediction_text='Emotion $ {}'.format(output))

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])
#
#     output = prediction[0]
#     return jsonify(output)



if __name__ == "__main__":
    app.run(debug=True)