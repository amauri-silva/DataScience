import pandas as pd
from dbconnection import insert_tb_aggtrade, postgres_connector

def agg_trade_csv(data_frame):
    df1 = data_frame
    df1.to_csv('dataset/output/agg_trade_solusdt.csv', mode='a', header=False, index=False, sep=',')
    
    # Save to CSV and run a bash process to load the original data on database
    # print("Opening connection to PostgreSQL DB...")
    # pg_connection  = postgres_connector.postgres_connector()
    # print("##########################:   {}".format(df1['event_time']))
    # insert_tb_aggtrade.insert_tb_aggtrade(pg_connection, df1)
    # print("Data saved successfully on tb_agg_trade")

def trade_csv(data_frame):
    df1 = data_frame
    df1.to_csv('dataset/output/trade_solusdt.csv', mode='a', header=False, index=False, sep=',')

def avg_price_csv(data_frame):
    df1 = data_frame
    df1.to_csv('dataset/output/avg_price_solusdt.csv', mode='a', header=False, index=False, sep=',')

def book_ticker_csv(data_frame):
    df1 = data_frame
    df1.to_csv('dataset/output/book_ticker_solusdt.csv', mode='a', header=False, index=False, sep=',')

def kline_interval_csv(data_frame):
    df1 = data_frame
    df1.to_csv('dataset/output/kline_interval_solusdt.csv', mode='a', header=False, index=False, sep=',')

def ticker_window_csv(data_frame):
    df1 = data_frame
    df1.to_csv('dataset/output/ticker_window_solusdt.csv', mode='a', header=False, index=False, sep=',')

def ticker_window_arr_csv(data_frame):
    df1 = data_frame
    df1.to_csv('dataset/output/ticker_window_arr_solusdt.csv', mode='a', header=False, index=False, sep=',')