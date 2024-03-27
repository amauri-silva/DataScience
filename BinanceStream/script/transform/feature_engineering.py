import pandas as pd
from config import constants
from load import store_data
from utils import convert_timestamp


def event_time_df(timestamp):
    """
    Transform the event time to a datetime format
    aggt_event_time_dt DATE,
	aggt_event_time_tm TIME,
    """

    full_date_time, only_date, only_time = convert_timestamp.get_date_time_from_timestamp(timestamp)

    df = pd.DataFrame({
        'event_time_dt': [only_date],
        'event_time_tm': [only_time]
    })    

    return df

def trade_time_df(timestamp):
    """
    Transform the event time to a datetime format
	aggt_trade_time_dt DATE,
	aggt_trade_time_tm TIME
    """

    full_date_time, only_date, only_time = convert_timestamp.get_date_time_from_timestamp(timestamp)

    df = pd.DataFrame({
        'trade_time_dt': [only_date],
        'trade_time_tm': [only_time]
    })    

    return df


def transform_agg_trade(json):
    print(json['data'])
    df = pd.DataFrame.from_dict(json['data'], orient='index')
    df_t = df.transpose()
    df2 = df_t.rename(columns=constants.COLUNMS_NAME_AGG_TRADE)
    
    # my_dict = df2.to_dict()
    # #my_dict2 = [my_dict[0]]
    # df_evt_time = event_time_df(df2['event_time'])
    # df_trd_time = trade_time_df(df2['trade_time'])
    # print(type(my_dict))

    # df2.append(df_evt_time, ignore_index=True)
    # df2.append(df_trd_time, ignore_index=True)
    # my_dict.append(df_evt_time)
    # print("back againnnnnnnnnnnnnnnnnnnnnnnnnnn")
    store_data.agg_trade_csv(df2)


def transform_trade(json):
    print(json['data'])
    df = pd.DataFrame.from_dict(json['data'], orient='index')
    df_t = df.transpose()
    df2 = df_t.rename(columns=constants.CONLUMS_NAME_TRADE)

    store_data.trade_csv(df2)


def transform_avg_price(json):
    print(json['data'])
    df = pd.DataFrame.from_dict(json['data'], orient='index')
    df_t = df.transpose()
    df2 = df_t.rename(columns=constants.CONLUMS_NAME_AVG_PRICE)

    store_data.avg_price_csv(df2)

def transform_book_ticker(json):
    print(json['data'])
    df = pd.DataFrame.from_dict(json['data'], orient='index')
    df_t = df.transpose()
    df2 = df_t.rename(columns=constants.CONLUMS_NAME_BOOK_TICKER)

    store_data.book_ticker_csv(df2)

def transform_kline_interval(json):
    print(json['data']['k'])
    df = pd.DataFrame.from_dict(json['data']['k'], orient='index')
    df_t = df.transpose()
    df2 = df_t.rename(columns=constants.CONLUMS_NAME_KLINE_INTERVAL)

    store_data.kline_interval_csv(df2)


def transform_ticker_window(json):
    print(json['data'])
    df = pd.DataFrame.from_dict(json['data'], orient='index')
    df_t = df.transpose()
    df2 = df_t.rename(columns=constants.CONLUMS_NAME_TICKER_WINDOW)

    store_data.ticker_window_csv(df2)