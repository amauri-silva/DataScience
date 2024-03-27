import json
import pandas as pd
import os
import json
import websocket
import pandas as pd
from csv import writer
from utils import stream_names
from config import constants
from transform import feature_engineering



# def data_frame_sttrade(json):
#     js = json.loads(json.text)
#     print(js)
#     df = pd.DataFrame(js, columns=_COLUMNS_NAME_STTRADE)

#     print(json)
#     return df


# def json_to_csvfile(csv_file, data_frame):
#     if(os.path.exists(csv_file) and os.stat(csv_file) != 0):

#         print(data_frame)
#         #write adding new lines
#         with open(csv_file, 'a', newline='') as object:
#             w_object = writer(object)
#             for index, row in data_frame.iterrows():
#                 w_object.writerow(row)
#             object.close()

#     else:
#         data_frame.to_csv(csv_file, sep=",", index=False)


def agg_trade_on_message(ws,wss_json_content):
    json_content = json.loads(wss_json_content)
    feature_engineering.transform_agg_trade(json_content)

def trade_on_message(ws, wss_json_content):
    json_content = json.loads(wss_json_content)
    print(json_content)
    
    feature_engineering.transform_trade(json_content)

def avg_price_on_message(ws, wss_json_content):
    json_content = json.loads(wss_json_content)
    
    feature_engineering.transform_avg_price(json_content)



def book_ticker_on_message(ws, wss_json_content):
    json_content = json.loads(wss_json_content)
    
    feature_engineering.transform_book_ticker(json_content)


def kline_interval_on_message(ws, wss_json_content):
    json_content = json.loads(wss_json_content)
    
    feature_engineering.transform_kline_interval(json_content)

def ticker_window_arr_on_message(ws, wss_json_content):
    json_content = json.loads(wss_json_content)
    
    feature_engineering.transform_agg_trade(json_content)

def ticker_window_on_message(ws, wss_json_content):
    json_content = json.loads(wss_json_content)
    
    feature_engineering.transform_ticker_window(json_content)


def trade_solusdt_on_message(ws, wss_json_content):
    json_content = json.loads(wss_json_content)
    
    feature_engineering.transform_agg_trade(json_content)




# ========================================================================================================
# ========================================================================================================

def run_websocket_agg_trade(symbol):

    url = stream_names.agg_trade(symbol)
    socket = "wss://stream.binance.com:9443/stream?streams="+url
    print(socket)
    ws = websocket.WebSocketApp(socket, on_message=agg_trade_on_message)
    ws.run_forever()

def run_websocket_trade(symbol):

    url = stream_names.trade(symbol)
    socket = "wss://stream.binance.com:9443/stream?streams="+url
    ws = websocket.WebSocketApp(socket, on_message=trade_on_message)
    # ws.run_forever()

def run_websocket_avg_price(symbol):

    url = stream_names.avg_price(symbol)
    socket = "wss://stream.binance.com:9443/stream?streams="+url
    print(socket)
    ws = websocket.WebSocketApp(socket, on_message=avg_price_on_message)
    ws.run_forever()

def run_websocket_book_tickek(symbol):

    url = stream_names.book_ticker(symbol)
    socket = "wss://stream.binance.com:9443/stream?streams="+url
    print(socket)
    ws = websocket.WebSocketApp(socket, on_message=book_ticker_on_message)
    ws.run_forever()

def run_websocket_kline_interval(symbol):

    url = stream_names.kline(symbol, interval="1s")
    socket = "wss://stream.binance.com:9443/stream?streams="+url
    print(socket)
    ws = websocket.WebSocketApp(socket, on_message=kline_interval_on_message)
    ws.run_forever()

def run_websocket_ticker_window(symbol):

    url = stream_names.ticker_window(symbol, window_size="1h")
    socket = "wss://stream.binance.com:9443/stream?streams="+url
    print(socket)
    ws = websocket.WebSocketApp(socket, on_message=ticker_window_on_message)
    ws.run_forever()

def run_websocket_ticker_window_arr(symbol):

    url = stream_names.ticker_window_arr(symbol)
    socket = "wss://stream.binance.com:9443/stream?streams="+url
    print(socket)
    ws = websocket.WebSocketApp(socket, on_message=ticker_window_arr_on_message)
    ws.run_forever()
