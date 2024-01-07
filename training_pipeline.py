# Import necessary libraries
from prophet import Prophet

import hopsworks

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

from sklearn.metrics import mean_absolute_error

DEBUG = 0

MODEL_DIR = "./gold_model"

API_KEY = "WrS8FeVgLISFnH7c.OHnfbWyBTWGM3hCwqa4oBfAElkpJ73Sq4UBXYeSTfpiRlTSBINBDadbvNSqhQpRj" # API of hopsworks
FG_VERSION = 10
FV_VERSION = 2

BATCH = 8
TEST_SPLIT = 5

PARAMETERS = ['fearandgreed', 'gold_open', 'oil_open', 'usd_open', 'plat_open']

# get historical price data from hopsworks
def get_data_from_hopsworks():
    project = hopsworks.login(api_key_value=API_KEY)
    fs = project.get_feature_store()

    price_fg = fs.get_feature_group(name="price", version=FG_VERSION)
    query = price_fg.select_all()
    feature_view = fs.get_or_create_feature_view(name="price",
                                  version=FV_VERSION,
                                  description="Read from price dataset",
                                  labels=["gold_high"],
                                  query=query)
    
    historical_price = price_fg.read(read_options={"use_hive": True})

    historical_price.sort_values(by=['date'], inplace=True)
    historical_price.reset_index(drop=True, inplace=True)

    return historical_price

def train_model(data):
    model = Prophet()

    for parameter in PARAMETERS:
        model.add_regressor(parameter)

    data_train = data.rename(columns={'date':'ds', 'gold_high':'y'})
    data_train = data_train.drop(data_train.tail(TEST_SPLIT).index)

    model.fit(data_train)
    return model

def test_model(model, data):
    data_test = data.rename(columns={'date':'ds', 'gold_high':'y'})
    dates = data['date']
    data_test = data_test.tail(TEST_SPLIT)

    x_test = data_test.drop(axis=1, labels=['y'])
    y_true = data_test['y']
    y_test = data_test['y']

    forecast = model.predict(x_test)
    y_test = forecast[['yhat']]

    mae = mean_absolute_error(y_true, y_test)
    accuracy = 100 * (1 - mae / y_true.mean())

    return accuracy

def save_model(model, accuracy):
    if os.path.isdir(MODEL_DIR) == False:
        os.mkdir(MODEL_DIR)

    # Save both our model and the confusion matrix to 'model_dir', whose contents will be uploaded to the model registry
    joblib.dump(model, MODEL_DIR + "/price_model.pkl")  

    project = hopsworks.login(api_key_value=API_KEY)
    fs = project.get_feature_store()

    mr = project.get_model_registry()

    wine_model = mr.python.create_model(
        name="gold_model", 
        metrics={"accuracy": accuracy},
        description="Gold Maximum Price Predictor"
    )

    # Upload the model to the model registry, including all files in 'model_dir'
    wine_model.save(MODEL_DIR)

if __name__ == '__main__':
    historical_price = get_data_from_hopsworks()
    if DEBUG == 1:
        print(historical_price)
    model = train_model(historical_price)
    accuracy = test_model(model, historical_price)
    save_model(model, accuracy)



