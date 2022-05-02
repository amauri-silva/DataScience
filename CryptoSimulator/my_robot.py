import time
from simple_robot import feature_eng, api_get, api_post, get_result, compute_quantity, how_much_i_have
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import  config
from sklearn.preprocessing import MinMaxScaler


def load_lstm_model():
    print("[INFO] Carregando modelo criado no notebook anterior ...")
    model = load_model(config.LSTM_MODEL_PATH)
    return model

def crypto_bot(token, ticker, valor_compra_venda):
    print("[INFO] Loop que checa o preço a cada minuto e toma as decisões de compra e venda ...")
    model = load_lstm_model()
    while True:
        # Pegando o OHLC dos últimos 500 minutos
        df = api_post('cripto_quotation', {'token': token, 'ticker': ticker})

        # Realizando a engenharia de features
        df = feature_eng(df)

        # Isolando a linha mais recente
        #index = int(np.argmax(df['time'])) -1
        #df_last = df[np.argmax(df['time'])]
        #df_last_2 = df.iloc[np.argmax(df['time'])]
        df_last = df.iloc[-1]

        # Calculando tendência, baseada no modelo linear criado
        mms = MinMaxScaler(feature_range=(0, 1))
        test = np.expand_dims(df_last, axis=0)
        stock_data_transf = mms.fit_transform(np.array(test).reshape(-1, 1))
        tendencia = model.predict(stock_data_transf)

        # A quantidade de cripto que será comprada/ vendida depende do valor_compra_venda e da cotação atual
        qtdade = compute_quantity(coin_value=df_last['close'], invest_value=valor_compra_venda, significant_digits=2)

        # Print do datetime atual
        print('-------------------')
        now = pd.to_datetime('now')
        print(f'{now}')

        if tendencia > 0.02:
            # Modelo detectou uma tendência positiva
            print(f"Tendência positiva de {str(tendencia)}")

            # Verifica quanto dinheiro tem em caixa
            qtdade_money = how_much_i_have('money', token)

            if qtdade_money > 0:
                # Se tem dinheiro, tenta comprar o equivalente a qtdade ou o máximo que o dinheiro permitir
                max_qtdade = compute_quantity(coin_value=df_last['close'], invest_value=qtdade_money, significant_digits=2)
                qtdade = min(qtdade, max_qtdade)

                # Realizando a compra
                print(f'Comprando {str(qtdade)} {ticker}')
                api_post('buy', payload={'token': token, 'ticker': ticker, 'quantity': qtdade})

        elif tendencia < -0.02:
            # Modelo detectou uma tendência negativa
            print(f"Tendência negativa de {str(tendencia)}")

            # Verifica quanto tem da moeda em caixa
            qtdade_coin = how_much_i_have(ticker, token)

            if qtdade_coin > 0:
                # Se tenho a moeda, vou vender!
                qtdade = min(qtdade_coin, qtdade)
                print(f'Vendendo {str(qtdade)} {ticker}')
                api_post('sell', payload={'token': token, 'ticker': ticker, 'quantity': qtdade})
        else:
            # Não faz nenhuma ação, espera próximo loop
            print(f"Tendência neutra de {str(tendencia)}. Nenhuma ação realizada")

        # Print do status após cada iteração
        print(api_post('status', payload={'token': token}))
        time.sleep(60)

if __name__ == "__main__":
    token = 'ba0b83cb57fcddf'
    ticker = 'DOGEUSDT'
    valor_compra_venda = 10

    crypto_bot(token, ticker, valor_compra_venda)