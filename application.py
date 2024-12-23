# flask, pandas, scikit-learn, pickle-mixin
# from conda_env.installers import pip
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from flask_cors import CORS, cross_origin

import pickle

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('Regression_Model.pkl', 'rb'))
data = pd.read_csv('Cleaned_Car_Data.csv')


@app.route('/')
def index():
    companies = data['company'].unique()
    sort_companies = np.sort(companies)

    car_models = data['name'].unique()
    sort_car = np.sort(car_models)

    years = data['year'].unique()
    sort_year = np.sort(years)[::-1]

    fuel_type = data['fuel_type'].unique()
    return render_template('index.html', companies=sort_companies, car_models=sort_car, years=sort_year,
                           fuel_type=fuel_type)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():

    company = request.form.get('company')

    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0], 2))
