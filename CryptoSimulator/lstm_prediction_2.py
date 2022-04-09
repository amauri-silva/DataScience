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

def download_dataset_file():
    os.path.join(config.DATA_FRAME_SAVE_PATH, "crypto_dataset")

def datetime_to_timestamp(date_time):
    dt_list = []

    for dt in date_time:
        times_tamp = time.mktime(datetime.strptime(str(dt), '%Y-%m-%d %H:%M:%S').timetuple())
        timestamp = int(times_tamp)
        dt_list.append(timestamp)

    return dt_list

def stock_data():
    print("[INFO] Obtendo os dados (dataset) ...")
    data_frame = pd.read_parquet(config.DATA_FRAME_PATH_DOWNLOAD)
    # Calculando qual a média de close dos próximos 10min
    data_frame['forward_average'] = data_frame[::-1]['close'].rolling(10).mean()[::-1].shift(-1)

    # Target será a diferença percentual do 'forward_average' com o 'close' atual
    data_frame['target'] = 100 * (data_frame['forward_average'] - data_frame['close']) / data_frame['close']

    #data_frame.head(20)
    #display(data_frame)
    # Outra possibilidade: target como a diferença entre o proximo minuto e o atual: data_frame['diff']= -data_frame['close'].diff(-1)
    return data_frame

def split_dataset(df):

    data_frame = feature_eng(df)
    print("[INFO] Separando os dados em treino/teste ...")
    #Separando usando data. Isso é importante, pois precisamos entender se os modelos criados em um tempo passado
    # continua sendo útil em um tempo futuro.
    test_treshold = '2021-06-01 00:00:00'

    train = data_frame[data_frame.index <= test_treshold]
    test = data_frame[data_frame.index > test_treshold]

    X_train = train.drop(columns=['target'])
    y_train = train['target']

    X_test = test.drop(columns=['target'])
    y_test = test['target']

    return X_train,y_train, X_test, y_test

def pre_precess_data():

    data_frame = pd.read_parquet(config.DATA_FRAME_PATH_DOWNLOAD)
    data_frame = data_frame.drop(columns=["symbol"])
    # Calculando qual a média de close dos próximos 10min
    data_frame['forward_average'] = data_frame[::-1]['close'].rolling(10).mean()[::-1].shift(-1)

    # Target será a diferença percentual do 'forward_average' com o 'close' atual
    data_frame['target'] = 100 * (data_frame['forward_average'] - data_frame['close']) / data_frame['close']

    feature_eng(data_frame)

    #datetime = np.array(data_frame['datetime'])
    datetime = datetime_to_timestamp(data_frame['datetime'])
    data_frame['datetime'] = datetime

    display(data_frame)
    #open_time = np.array(data_frame[0])
    #data_frame['open_time'] = open_time

    #data_frame.drop(columns=['target'])

    mms = MinMaxScaler()
    stock_data_transf = mms.fit_transform(data_frame)
    training_size = round(len(stock_data_transf) * 0.80)

    train_data = stock_data_transf[:training_size]
    test_data = stock_data_transf[training_size:]

    #X_train = train_data.drop(columns=['target'])
    X_train = np.delete(train_data, 9, 0)
    #y_train = train_data['target']
    y_train = train_data[9]

    #X_test = test_data.drop(columns=['target'])
    X_test = np.delete(train_data, 9, 0)
    #y_test = test_data['target']
    y_test = test_data[9]

    return X_train, y_train, X_test, y_test

def traing_model(X_train,y_train, X_test, y_test):
    print("[INFO] Treinando o modelo baseado em LSTM ...")
    model = Sequential()
    model.add(LSTM(units=1000, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=1000))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=2)

    # Salvando o modelo em um arquivo pickle para ser utilizado nas etapas seguintes
    filename = 'model_dummy.pickle'
    pickle.dump(model, open(filename, 'wb'))


#--------------------------------------------------------------------------------------

if __name__ == '__main__':
    data_frame = stock_data()
    #display(data_frame)
    #X_train, y_train, X_test, y_test = split_dataset(data_frame)
    X_train, y_train, X_test, y_test = pre_precess_data()
    traing_model(X_train, y_train, X_test, y_test)
    # display(train, test)