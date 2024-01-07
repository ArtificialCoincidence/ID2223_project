import os

import requests, csv, pytz, json
from datetime import datetime

import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import hopsworks

FEARANDGREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata" # url of graph data of historical CNN fear and greed index
START_DATE = '2020-12-11'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'}

HISTORY_FEARANDGREED = './history_data/history_fearandgreed.csv'
HISTORY_GOLD         = './history_data/history_gold.csv'
HISTORY_OIL          = './history_data/history_oil.csv'
HISTORY_USD          = './history_data/history_usd.csv'
HISTORY_PLATINUM     = './history_data/history_platinum.csv'

API_KEY = "WrS8FeVgLISFnH7c.OHnfbWyBTWGM3hCwqa4oBfAElkpJ73Sq4UBXYeSTfpiRlTSBINBDadbvNSqhQpRj" # API of hopsworks
FG_VERSION = 10

FEARANDGREED_COLUMNS = ['date', 'fearandgreed']
GOLD_COLUMNS = ['date', 'gold_open', 'gold_high', 'gold_low', 'gold_close', 'gold_adjclose', 'gold_volume']
OIL_COLUMNS  = ['date',  'oil_open',  'oil_high',  'oil_low',  'oil_close',  'oil_adjclose',  'oil_volume']
USD_COLUMNS  = ['date',  'usd_open',  'usd_high',  'usd_low',  'usd_close',  'usd_adjclose',  'usd_volume']
PLAT_COLUMNS = ['date', 'plat_open', 'plat_high', 'plat_low', 'plat_close', 'plat_adjclose', 'plat_volume']

PARAMETERS = ['date', 'fearandgreed', 'gold_open', 'oil_open', 'usd_open', 'plat_open']
TARGET = ['gold_high']

def upload_to_hopsworks(data, parameter_keys):
    project = hopsworks.login(api_key_value=API_KEY)
    fs = project.get_feature_store()

    price_fg = fs.get_or_create_feature_group(
        name="price",
        version=FG_VERSION,
        primary_key=parameter_keys, 
        description="Price dataset")
    price_fg.insert(data)

def get_history_fearandgreed():
    r = requests.get("{}/{}".format(FEARANDGREED_URL, START_DATE), headers=HEADERS)
    data = r.json()

    fg_data = data['fear_and_greed_historical']['data']

    fear_greed_values = {}

    with open(HISTORY_FEARANDGREED, 'x', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Date', 'Fear Greed'])

        for data in fg_data:
            dt = datetime.fromtimestamp(data['x'] / 1000, tz=pytz.utc)
            dt = dt.strftime("%Y/%m/%d")
            fear_greed_values[dt] = float(data['y'])
            writer.writerow([dt, float(data['y'])])

def get_history_data():
    # read history data from files
    history_fearandgreed = pd.read_table(HISTORY_FEARANDGREED, sep=',', names=FEARANDGREED_COLUMNS, skiprows=1, header=None)
    history_gold         = pd.read_table(HISTORY_GOLD, sep=',', names=GOLD_COLUMNS, skiprows=1, header=None)
    history_oil          = pd.read_table(HISTORY_OIL, sep=',', names=OIL_COLUMNS, skiprows=1, header=None)
    history_usd          = pd.read_table(HISTORY_USD, sep=',', names=USD_COLUMNS, skiprows=1, header=None)
    history_platinum     = pd.read_table(HISTORY_PLATINUM, sep=',', names=PLAT_COLUMNS, skiprows=1, header=None)
    
    # delete lines with nan value in open price in historical data
    history_gold.dropna(subset=['gold_open'], inplace=True)
    history_oil.dropna(subset=['oil_open'], inplace=True)
    history_usd.dropna(subset=['usd_open'], inplace=True)
    history_platinum.dropna(subset=['plat_open'], inplace=True)

    # we do not need the volume column for this project
    history_gold.drop(axis=1, labels=['gold_volume'], inplace=True)
    history_platinum.drop(axis=1, labels=['plat_volume'], inplace=True)
    history_oil.drop(axis=1, labels=['oil_volume'], inplace=True)
    history_usd.drop(axis=1, labels=['usd_volume'], inplace=True)

    # reverse fear and greed index and delete the repleting data
    history_fearandgreed = history_fearandgreed[::-1].reset_index(drop=True)
    history_fearandgreed.drop(index=0, inplace=True)

    history_usd = history_usd[::-1].reset_index(drop=True)

    # select the data of sharing dates and merge the dataframes
    history_data_sorted = history_fearandgreed
    history_data_sorted = history_data_sorted.merge(right=history_gold, how='inner', on='date') # integrate two datasets to one
    history_data_sorted = history_data_sorted.merge(right=history_oil, how='inner', on='date')
    history_data_sorted = history_data_sorted.merge(right=history_usd, how='inner', on='date')
    history_data_sorted = history_data_sorted.merge(right=history_platinum, how='inner', on='date')

    history_data_sorted.iloc[:, 2:26] = history_data_sorted.iloc[:, 2:26].astype('float')

    history_data_selected = history_data_sorted[PARAMETERS + TARGET]

    return history_data_selected



if __name__ == '__main__':

    if os.path.exists(HISTORY_FEARANDGREED) == 0:
        get_history_fearandgreed()

    history_data = get_history_data()

    print(history_data)
    
    upload_to_hopsworks(history_data, PARAMETERS)