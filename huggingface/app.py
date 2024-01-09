from PIL import Image
import gradio as gr
import requests
import hopsworks
import joblib
import pandas as pd
import prophet
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import requests, pytz

PRICE_DATA = ['gold_open', 'oil_open', 'usd_open', 'plat_open']
PRICE_OPEN_URL = {"gold_open": "https://finance.yahoo.com/quote/GC%3DF",
                  "oil_open" : "https://finance.yahoo.com/quote/CL%3DF",
                  "usd_open" : "https://finance.yahoo.com/quote/DX-Y.NYB",
                  "plat_open": "https://finance.yahoo.com/quote/PL%3DF"}
FEARANDGREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata" # url of graph data of historical CNN fear and greed index
START_DATE = '2023-12-11'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'}

def get_date():
    now = datetime.now()
    date_today = now.strftime("%Y/%m/%d")
    return date_today

def get_date_tomorrow():
    tomorrow = datetime.now() + timedelta(1)
    date_tomorrow = tomorrow.strftime("%Y/%m/%d")
    return date_tomorrow

def get_model_from_hopsworks():
    project = hopsworks.login()
    fs = project.get_feature_store()

    mr = project.get_model_registry()
    model = mr.get_best_model("gold_model", "accuracy", "max")
    model_dir = model.download()
    model = joblib.load(model_dir + "/gold_model.pkl")
    print("Model downloaded")

    return model

def predict_with_model(data, model):
    data = data.rename(columns={'date':'ds'})
    forecast = model.predict(data)
    prediction = forecast[['yhat']]
    return prediction['yhat'][0]

def get_fearandgreed(date):
    r = requests.get("{}/{}".format(FEARANDGREED_URL, START_DATE), headers=HEADERS)
    data = r.json()

    fg_data = data['fear_and_greed_historical']['data']

    fear_greed_values = {}

    for data in fg_data:
        dt = datetime.fromtimestamp(data['x'] / 1000, tz=pytz.utc)
        dt = dt.strftime("%Y/%m/%d")
        fear_greed_values[dt] = float(data['y'])

    return fear_greed_values[date]

def predict_data_today():

    price_today = {}
    for price in PRICE_DATA:
        response = requests.get(url=PRICE_OPEN_URL[price], headers=HEADERS)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text)
            price_open_str = soup.find('td', {'data-test': 'OPEN-value'}).text
            price_open_str = price_open_str.replace(',', '')
            price_open_num = float(price_open_str)
            price_today[price] = [price_open_num]
        else:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")

    date = get_date()
    fearandgreed = get_fearandgreed(date)

    data_today = pd.DataFrame(price_today)
    data_today.insert(0, 'fearandgreed', fearandgreed)
    data_today.insert(0, 'date', date)
    data_today.iloc[:, 1:5] = data_today.iloc[:, 1:5].astype('float')

    print(data_today)

    global model
    prediction = predict_with_model(data_today, model)

    return [price_today['gold_open'][0], price_today['usd_open'][0], price_today['oil_open'][0], price_today['plat_open'][0], prediction]

def predict_data_tomorrow(fearandgreed, gold_open, usd_open, oil_open, plat_open):
    data = pd.DataFrame()
    data['fearandgreed'] = [fearandgreed]
    data['gold_open'] = [gold_open]
    data['usd_open'] = [usd_open]
    data['oil_open'] = [oil_open]
    data['plat_open'] = [plat_open]

    date = get_date_tomorrow()
    data.insert(0, 'date', date)

    global model
    prediction = predict_with_model(data, model)

    return prediction

def show_image():
    project = hopsworks.login()
    dataset_api = project.get_dataset_api()

    dataset_api.download("Resources/images/gold_price_prediction_recent.png", overwrite=True)

show_image()
model = get_model_from_hopsworks()

with gr.Blocks() as demo:
    title = gr.Markdown(""" # Highest Gold Price Prediction """)
    with gr.Tabs():
        with gr.TabItem("Daily highest gold price prediction"):
            txt_gold_open = gr.Textbox(value="", label="gold open price")
            txt_usd_open  = gr.Textbox(value="", label="usd open price")
            txt_oil_open  = gr.Textbox(value="", label="crude oil open price")
            txt_plat_open = gr.Textbox(value="", label="platinum open price")
            txt_gold_high = gr.Textbox(value="", label="Predicted highest gold price today")
            btn = gr.Button(value="Press me!")
            btn.click(predict_data_today, outputs=[txt_gold_open, txt_usd_open, txt_oil_open, txt_plat_open, txt_gold_high])
        with gr.TabItem("Predict highest gold price tomorrow with your preferred values"):
            txt_fearandgreed = gr.Textbox(value="", label="Enter fear and greed index")
            txt_gold_open = gr.Textbox(value="", label="Enter gold open price")
            txt_usd_open  = gr.Textbox(value="", label="Enter usd open price")
            txt_oil_open  = gr.Textbox(value="", label="Enter crude oil open price")
            txt_plat_open = gr.Textbox(value="", label="Enter platinum open price")
            txt_gold_high = gr.Textbox(value="", label="Predicted highest gold price tomorrow ")
            btn = gr.Button(value="Press me!")
            btn.click(predict_data_tomorrow, inputs=[txt_fearandgreed, txt_gold_open, txt_usd_open, txt_oil_open, txt_plat_open], outputs=[txt_gold_high])
        with gr.TabItem("Recent predicted and true values of highest gold price"):
            image = gr.Image("gold_price_prediction_recent.png", elem_id="recent gold price prediction")
            btn = gr.Button(value="Press me to see the data!")
            btn.click(show_image)

if __name__ == "__main__":
    demo.launch(debug=True, share=True)