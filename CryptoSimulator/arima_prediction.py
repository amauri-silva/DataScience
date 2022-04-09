import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import statsmodels.api as sm
import pickle
from simple_robot import feature_eng
import requests
from IPython.display import display
import config
import os
from datetime import datetime
import time
#=============================

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def datetime_to_timestamp(date_time):
    dt_list = []

    for dt in date_time:
        times_tamp = time.mktime(datetime.strptime(str(dt), '%Y-%m-%d %H:%M:%S').timetuple())
        timestamp = int(times_tamp)
        dt_list.append(timestamp)

    return dt_list

def pre_precess_data():

    data_frame = pd.read_parquet(config.DATA_FRAME_PATH_DOWNLOAD)
    data_frame = data_frame.drop(columns=["symbol"])
    # Calculando qual a média de close dos próximos 10min
    data_frame['forward_average'] = data_frame[::-1]['close'].rolling(10).mean()[::-1].shift(-1)

    # Target será a diferença percentual do 'forward_average' com o 'close' atual
    data_frame['target'] = 100 * (data_frame['forward_average'] - data_frame['close']) / data_frame['close']

    feature_eng(data_frame)

    #datetime = np.array(data_frame['datetime'])
    #datetime = datetime_to_timestamp(data_frame['datetime'])
    #data_frame['datetime'] = datetime

    display(data_frame)

    return data_frame['target']

def arima(df):
    # 1,1,2 ARIMA Model
    model = ARIMA(df, order=(1, 1, 2))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())


if __name__ == "__main__":
    df = pre_precess_data()
    arima(df)