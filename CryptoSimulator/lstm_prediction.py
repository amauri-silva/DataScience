from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional

from IPython.display import display


# https://www.analyticsvidhya.com/blog/2021/05/stock-price-prediction-and-forecasting-using-stacked-lstm/


def stock_data():
    df = pd.read_parquet('https://drive.google.com/u/0/uc?id=17c2r9qbnsxPVxaYukrp6vhTY-CQy8WZa&export=download')
    # Calculando qual a média de close dos próximos 10min
    df['forward_average'] = df[::-1]['close'].rolling(10).mean()[::-1].shift(-1)

    # Target será a diferença percentual do 'forward_average' com o 'close' atual
    df['target'] = 100 * (df['forward_average'] - df['close']) / df['close']

    #df.head(20)
    #display(df)
    # Outra possibilidade: target como a diferença entre o proximo minuto e o atual: df['diff']= -df['close'].diff(-1)
    return df

def pre_precess_data(data):
    #
    mms = MinMaxScaler()
    stock_data_transf = mms.fit_transform(data)
    training_size = round(len(stock_data_transf) * 0.80)

    train_data = stock_data_transf[:training_size]
    test_data = stock_data_transf[training_size:]


def lstm_implmentation(train_seq, train_label, test_seq, test_label, MMS, gstock_data):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(train_seq.shape[1], train_seq.shape[2])))

    model.add(Dropout(0.1))
    model.add(LSTM(units=50))
    model.add(Dense(2))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    model.summary()

    model.fit(train_seq, train_label, epochs=80, validation_data=(test_seq, test_label), verbose=1)
    test_predicted = model.predict(test_seq)
    test_inverse_predicted = MMS.inverse_transform(test_predicted)

    # Merging actual and predicted data for better visualization
    gs_slic_data = pd.concat([gstock_data.iloc[-202:].copy(),
                              pd.DataFrame(test_inverse_predicted, columns=['open_predicted', 'close_predicted'],
                                           index=gstock_data.iloc[-202:].index)], axis=1)
    gs_slic_data[['open', 'close']] = MMS.inverse_transform(gs_slic_data[['open', 'close']])
    gs_slic_data.head()


# --------------------------------------------------
stock_data()
