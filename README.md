# ID2223_project
This is a project using open prices of gold, platinum, USD, crude oil and CNN fear and greed index to predict highest price of gold daily.
# Author 
Ruijia Dai, Zahra Khorasani Zavareh
# Huggingface link
https://huggingface.co/spaces/ArtificialCoincidence/Daily_Gold_Highest_Price_Prediction \
User can:
1. See prediction of today's highest gold price (if market opens today and CNN fear and greed value is available)
2. Make prediction of highest gold price with preferred value of open prices
3. See recent predictions and true values of daily highest gold price
# How to run
1. Run eda_and_backfill_featuregroup.py
2. Run training_pipeline.py
3. Run feature_inference_training_pipeline_daily.py daily
# Data resources
gold price          : https://finance.yahoo.com/quote/GC%3DF?p=GC%3DF \
platinum price      : https://finance.yahoo.com/quote/PL%3DF?p=PL%3DF \
USD price           : https://finance.yahoo.com/quote/DX-Y.NYB?p=DX-Y.NYB \
crude oil price     : https://finance.yahoo.com/quote/CL%3DF?p=CL%3DF \
fear and greed index: https://edition.cnn.com/markets/fear-and-greed
# Learning algorithm
Facebook Prophet
# Implementation
1. Fetch data of latest 3 years from data resources (eda_and_backfill_featuregroup.py)
2. Select available data and upload to feature group (eda_and_backfill_featuregroup.py)
3. Train model on train dataset and test model on test dataset(5 days' available data latest <=> last 5 rows of data), then save model (training_pipeline.py)
4. Daily: add yesterday's data to feature group (feature_inference_training_pipeline_daily.py)
5. Daily: predict today's highest gold price with saved model (feature_inference_training_pipeline_daily.py)
6. Daily: retrain model on latest data in feature group (feature_inference_training_pipeline_daily.py)
# Result
Model evaluation method: MSE(highest gold price true value, highest gold price prediction) \
Recent predictions and true values (prediction: red--, true value: green- -): ![](./gold_price_prediction_recent.png) \
Automated daily run test: from 2023.12.28 to 2024.01.07 run successfully, can skip insertion and prediction when any necessary data is
unavailable \

