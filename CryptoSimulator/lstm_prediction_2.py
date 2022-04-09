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
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report


def download_dataset_file():
    os.path.join(config.DATA_FRAME_SAVE_PATH, "crypto_dataset")

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
    dataX.append(a)
    dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

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
    print(type(data_frame))

    #Remove linhas no qual contem valores NA (em uma pré analize foi contabilizada 39 linhas)
    data_frame = data_frame.dropna()

    display(data_frame)
    #open_time = np.array(data_frame[0])
    #data_frame['open_time'] = open_time

    #data_frame.drop(columns=['target'])

    mms = MinMaxScaler(feature_range=(0,1))
    stock_data_transf = mms.fit_transform(np.array(data_frame).reshape(-1,1))
    #training_size = int(len(stock_data_transf) * 0.80)
    training_size = round(len(stock_data_transf) * 0.80)

    train_data = stock_data_transf[:training_size]
    test_data = stock_data_transf[training_size:]

    #INICIO V1.0--------------------------------------------------------------------------------------------------------
    #X_train = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
    #X_test = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)


    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    #FIM ---------------------------------------------------------------------------------------------------------------

    # #X_train = train_data.drop(columns=['target'])
    # X_train = np.delete(train_data, 9, 0)
    # #y_train = train_data['target']
    # y_train = train_data[9]
    #
    # #X_test = test_data.drop(columns=['target'])
    # X_test = np.delete(train_data, 9, 0)
    # #y_test = test_data['target']
    # y_test = test_data[9]

    return X_train, y_train, X_test, y_test, mms

def traing_model(X_train, y_train, X_test, y_test, mms):
    print("[INFO] Treinando o modelo baseado em LSTM ...")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    #INICIO V1.0--------------------------------------------------------------------------------------------------------
    model.fit(X_train, X_test, epochs=1, batch_size=1, verbose=2)
    model.summary()
    # 8 - Treinamento do Modelo (train the head of the network)
    # print("[INFO] training head...")
    # model.fit(
    #     #steps_per_epoch=len(trainX) // BS,
    #     validation_data=(X_test, y_test),
    #     validation_steps=len(X_test))

    #FIM ---------------------------------------------------------------------------------------------------------------
    print("[INFO] Executando a predição do X_train ...")
    train_predict = model.predict(X_train)

    print("[INFO] Executando a predição do X_test ...")
    test_predict = model.predict(X_test)

    train_predict = mms.inverse_transform(train_predict)
    print("[INFO] Executando a predição do train_predict ...")

    test_predict = mms.inverse_transform(test_predict)
    print("[INFO] Executando a predição do test_predict ...")

    math.sqrt(mean_squared_error(y_train, train_predict))

    # 9 - Executa Testes de Predição
    print("[INFO] evaluating network...")
    predIdxs = model.predict(X_test, batch_size=1)

    # show a nicely formatted classification report
    #print(classification_report(X_test.argmax(axis=1), predIdxs))
    print("[INFO] printing predIdxs ...")
    print(predIdxs)
    # ----------------------------------------------------------------------------------------------------------------------
    # Salvando o modelo em um arquivo pickle para ser utilizado nas etapas seguintes
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))


#--------------------------------------------------------------------------------------

if __name__ == '__main__':
    data_frame = stock_data()
    #display(data_frame)
    #X_train, y_train, X_test, y_test = split_dataset(data_frame)
    X_train, y_train, X_test, y_test, mms = pre_precess_data()
    traing_model(X_train, y_train, X_test, y_test, mms)
    # display(train, test)