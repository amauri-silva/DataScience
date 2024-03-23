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



def data_frame_sttrade(json):
    js = json.loads(json.text)
    print(js)
    df = pd.DataFrame(js, columns=_COLUMNS_NAME_STTRADE)

    print(json)
    return df


def json_to_csvfile(csv_file, data_frame):
    if(os.path.exists(csv_file) and os.stat(csv_file) != 0):

        print(data_frame)
        #write adding new lines
        with open(csv_file, 'a', newline='') as object:
            w_object = writer(object)
            for index, row in data_frame.iterrows():
                w_object.writerow(row)
            object.close()

    else:
        data_frame.to_csv(csv_file, sep=",", index=False)

def on_message(ws, wss_json_content):
    json_content = json.loads(wss_json_content)
    
    feature_engineering.transform_agg_trade(json_content)

    
    
def run_websocket_agg_trade(symbol):

    url = stream_names.agg_trade(symbol)
    socket = "wss://stream.binance.com:9443/stream?streams="+url
    print(socket)
    #socket

    #websocket.enableTrace(True)
    ws = websocket.WebSocketApp(socket, on_message=on_message)
    ws.run_forever()
