import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import joblib


app = Flask(__name__)
ml_model=joblib.load('rf_model.pkl')
print("Model Type:", type(ml_model))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    output=ml_model.predict(data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    form = request.form

    # Numeric inputs
    CreditScore = int(form['CreditScore'])
    Age = int(form['Age'])
    Tenure = int(form['Tenure'])
    Balance = float(form['Balance'])
    NumOfProducts = int(form['NumOfProducts'])
    HasCrCard = int(form['HasCrCard'])
    IsActiveMember = int(form['IsActiveMember'])
    EstimatedSalary = float(form['EstimatedSalary'])
    Gender_Male = int(form['Gender_Male'])

    # Geography handling (one-hot encoding)
    geo = form['Geography']
    Geography_Germany = 1 if geo == 'Germany' else 0
    Geography_Spain = 1 if geo == 'Spain' else 0

    # IMPORTANT: order must match training data
    input_data = [[
        CreditScore,
        Age,
        Tenure,
        Balance,
        NumOfProducts,
        HasCrCard,
        IsActiveMember,
        EstimatedSalary,
        Geography_Germany,
        Geography_Spain,
        Gender_Male
    ]]

    prediction = ml_model.predict(input_data)

    return render_template(
        'home.html',
        prediction_text=f'The predicted output is: {prediction[0]}'
    )


if __name__ == "__main__":
    app.run(debug=True)