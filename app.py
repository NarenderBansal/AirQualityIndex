#comments are added.
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, url_for
import pickle

# load the model from disk
app = Flask(__name__)
model = pickle.load(open('Randonforest.pkl','rb'))

@app.route('/')
def home():
	return render_template('home.html')
	

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features)
    data=final_features.reshape(1,-1)
    predictions=model.predict(data)
    output = round(predictions[0], 2)
    return render_template('home.html', prediction_text='AQI for Delhi {}'.format(output))
	
if __name__ == '__main__':
    app.run(debug=True)
