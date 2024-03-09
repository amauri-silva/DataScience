import pandas as pd

def csv(data_frame):
    df1 = data_frame
    df1.to_csv('agg_trade_solusdt.csv', mode='a', header=False, index=False, sep=',')