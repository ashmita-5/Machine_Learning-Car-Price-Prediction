# app/routes.py
from flask import Flask, render_template, request, after_this_request
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

import pickle  # To load the model

app = Flask(__name__)

    # Load the trained model
loaded_data = pickle.load(open('car-prediction.pkl','rb'))

model = loaded_data['model']
scaler = loaded_data['scaler']
engine = loaded_data['engine']
max_power = loaded_data['max_power']
mileage = loaded_data['mileage']

@app.route('/',methods=['GET'])
#@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():

    if request.method == 'POST':
    # Extract user input and preprocess data
        engine = float(request.form['engine'])
        max_power = float(request.form['max_power'])
        mileage = float(request.form['mileage'])


    # Prepare input data for prediction
    input_data = np.array([[engine, max_power, mileage]])

    #Scaled the input data using the trained scaler
    sample_scaled = scaler.transform(input_data)

    # Make predictions
    predicted_price = model.predict(sample_scaled)[0]

    return render_template('index.html', predicted_price = np.exp(predicted_price))

if __name__ == '__main__':
    app.run(host="0.0.0.0:5000", debug=True)
