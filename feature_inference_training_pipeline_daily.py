# schedule: this program should be run daily after the stock market open
# function: 1. add yesterday's data to feature group
#           2. predict today's highest gold price with open prices
#           3. retrain model on the new data of feature group
# 
import modal
from datetime import datetime

import requests, pytz
from bs4 import BeautifulSoup
import time
import os
import joblib

import hopsworks

import pandas as pd

from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

DEBUG = 1

INSTALL_PACKAGE = ['cmdstanpy==1.0.4', 'datetime', 'requests', 'pytz', 'bs4', 'joblib', 'hopsworks', 'pandas', 'matplotlib', 'prophet==1.1.4', 'scikit-learn']

MODEL_DIR = "./gold_model"
TEST_SPLIT = 5

FG_VERSION = 10
FG_PRED_VERSION = 1
FV_VERSION = 2

FEARANDGREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata" # url of graph data of historical CNN fear and greed index
START_DATE = '2023-12-11'
PRICE_URL = {"gold_open": "https://finance.yahoo.com/quote/GC%3DF/history?p=GC%3DF",
             "oil_open" : "https://finance.yahoo.com/quote/CL%3DF/history?p=CL%3DF",
             "usd_open" : "https://finance.yahoo.com/quote/DX-Y.NYB/history?p=DX-Y.NYB",
             "plat_open": "https://finance.yahoo.com/quote/PL%3DF/history?p=PL%3DF"}
CLASS_NAME = 'BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'}

PRICE_DATA = ['gold_open', 'oil_open', 'usd_open', 'plat_open']
PARAMETERS = ['fearandgreed', 'gold_open', 'oil_open', 'usd_open', 'plat_open']
gold_high = 0
price_date = {}

stub = modal.Stub("Price_feature_pipeline_daily")
hopsworks_image = modal.Image.debian_slim().pip_install(INSTALL_PACKAGE)

@stub.function(image=hopsworks_image, schedule=modal.Cron("10 14 * * *"), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
def modal_function():
    from datetime import datetime
    import requests, pytz
    from bs4 import BeautifulSoup
    import time
    import os
    import joblib
    import hopsworks
    import pandas as pd
    from prophet import Prophet
    process_data_yesterday()

def parse_data(soup):
    data_yesterday      = soup.find('tr', class_=CLASS_NAME).find_next('tr', class_=CLASS_NAME)
    price_date      = data_yesterday.find_next('td')
    price_open      = price_date.find_next('td')
    price_high      = price_open.find_next('td')
    price_low       = price_high.find_next('td')
    price_close     = price_low.find_next('td')
    price_adjclose  = price_close.find_next('td')
    price_volume    = price_adjclose.find_next('td')
    return [price_date.text, price_open.text, price_high.text]

def get_fearandgreed(date):
    r = requests.get("{}/{}".format(FEARANDGREED_URL, START_DATE), headers=HEADERS)
    data = r.json()

    if r.status_code == 200:
        fg_data = data['fear_and_greed_historical']['data']

        fear_greed_values = {}

        for data in fg_data:
            dt = datetime.fromtimestamp(data['x'] / 1000, tz=pytz.utc)
            dt = dt.strftime("%Y/%m/%d")
            fear_greed_values[dt] = float(data['y'])

        return fear_greed_values[date]
    else:
        print(f"Failed to retrieve the webpage. Status code: {r.status_code}")
        return -1


def get_prices_yesterday():
    price_data = {}
    for price in PRICE_DATA:
        response = requests.get(url=PRICE_URL[price], headers=HEADERS)
        # Send a GET request to the URL

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text)
            [price_date_str, price_open_str, price_high_str] = parse_data(soup)
            print(f"{price} price open: {price_open_str}")

            date = datetime.strptime(price_date_str, "%b %d, %Y")
            # Format the datetime object as a string in the desired format
            price_date[price] = date.strftime("%Y/%m/%d")

            price_open_str = price_open_str.replace(',', '')
            if price_open_str == '-':
                print("Open price invalid, not processing data yesterday!")
                return -1
            price_open_num = float(price_open_str)
            price_data[price] = [price_open_num]

            if price == "gold_open":
                global gold_high
                price_high_str = price_high_str.replace(',', '')
                price_high_num = float(price_high_str)
                gold_high = price_high_num
                
        else:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")

    return price_data

def check_date():
    if (price_date['gold_open'] != price_date['oil_open']):
        return -1
    if (price_date['plat_open'] != price_date['usd_open']):
        return -1
    if (price_date['gold_open'] != price_date['usd_open']):
        return -1
    return 0

def get_data_from_hopsworks():
    project = hopsworks.login()
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

def plot_prediction():
    project = hopsworks.login()
    fs = project.get_feature_store()

    prediction_fg = fs.get_feature_group(name="price_prediction", version=FG_PRED_VERSION)
    query = prediction_fg.select_all()
    historical_prediction = prediction_fg.read(read_options={"use_hive": True})

    historical_prediction.sort_values(by=['date'], inplace=True)
    historical_prediction.reset_index(drop=True, inplace=True)

    plt.plot(historical_prediction['date'], historical_prediction['pred'], 'r--', historical_prediction['date'], historical_prediction['true'], 'g-')
    plt.title('Recent predicted and true values of highest gold price.')
    plt.xlabel('date')
    plt.ylabel('gold_high')
    plt.savefig('./gold_price_prediction_recent.png')
    project = hopsworks.login()
    dataset_api = project.get_dataset_api()
    dataset_api.upload("./gold_price_prediction_recent.png", "Resources/images", overwrite=True)

def upload_data_yesterday_to_hopsworks(data_yesterday):

    project = hopsworks.login()
    fs = project.get_feature_store()

    price_fg = fs.get_feature_group(name="price",version=FG_VERSION)
    price_fg.insert(data_yesterday)

def upload_data_prediction_to_hopsworks(data_prediction):
    project = hopsworks.login()
    fs = project.get_feature_store()

    price_fg = fs.get_or_create_feature_group(name="price_prediction",
                                              primary_key=['date'],
                                              version=FG_PRED_VERSION, 
                                              description="Price prediction dataset")
    price_fg.insert(data_prediction)

def get_model_from_hopsworks():
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_best_model("gold_model", "accuracy", "max")
    model_dir = model.download()
    model = joblib.load(model_dir + "/gold_model.pkl")

    return model

def predict_with_model(data, model):
    data = data.rename(columns={'date':'ds'})
    forecast = model.predict(data)
    prediction = forecast[['yhat']]
    return prediction['yhat'][0]

def train_model(data):
    model = Prophet()

    for parameter in PARAMETERS:
        model.add_regressor(parameter)

    data_train = data.rename(columns={'date':'ds', 'gold_high':'y'})
    data_train = data_train.drop(data_train.tail(TEST_SPLIT).index)
    if DEBUG == 1:
        print(data_train)

    model.fit(data_train)

    return model

def save_model(model, accuracy):
    if os.path.isdir(MODEL_DIR) == False:
        os.mkdir(MODEL_DIR)

    # Save both our model and the confusion matrix to 'model_dir', whose contents will be uploaded to the model registry
    joblib.dump(model, MODEL_DIR + "/gold_model.pkl")  

    project = hopsworks.login()
    fs = project.get_feature_store()

    mr = project.get_model_registry()

    wine_model = mr.python.create_model(
        name="gold_model",
        metrics={"accuracy": accuracy},
        description="Gold Maximum Price Predictor"
    )

    # Upload the model to the model registry, including all files in 'model_dir'
    wine_model.save(MODEL_DIR)

def test_model(model, data):
    data_test = data.rename(columns={'date':'ds', 'gold_high':'y'})
    dates = data['date']

    x_test = data_test.drop(axis=1, labels=['y'])
    y_true = data_test['y']
    y_test = data_test['y']

    forecast = model.predict(x_test)
    y_test = forecast[['yhat']]

    mae = mean_absolute_error(y_true, y_test)
    accuracy = 100 * (1 - mae / y_true.mean())

    return accuracy


def process_data_yesterday():

    price_prediction = {}
    prices = get_prices_yesterday()
    if prices == -1:
        return -1

    if check_date() != 0:
        print("Price dates not the same, not adding the data to feature group!")
    else:
        # feature pipeline
        fearandgreed = get_fearandgreed(price_date['gold_open'])
        if fearandgreed == -1:
            return -1
        data_yesterday = pd.DataFrame(prices)
        data_yesterday.insert(0, 'gold_high', gold_high)
        data_yesterday.insert(0, 'fearandgreed', fearandgreed)
        data_yesterday.insert(0, 'date', price_date['gold_open'])
        data_yesterday.iloc[:, 1:6] = data_yesterday.iloc[:, 1:6].astype('float')

        print(data_yesterday)

        upload_data_yesterday_to_hopsworks(data_yesterday)

        # inference pipeline
        model = get_model_from_hopsworks()
        prediction = predict_with_model(data_yesterday, model)

        price_prediction['date'] = [price_date['gold_open']]
        price_prediction['pred'] = [prediction]
        price_prediction['true'] = [gold_high]

        price_prediction = pd.DataFrame(price_prediction)
        upload_data_prediction_to_hopsworks(price_prediction)

        # training pipeline
        data_history = get_data_from_hopsworks()
        model_retrained = train_model(data_history)
        accuracy = test_model(model, data_history)
        save_model(model_retrained, accuracy)

        data_prediction = plot_prediction()



if __name__ == "__main__":

    modal.runner.deploy_stub(stub)
    with stub.run():
        modal_function.remote()