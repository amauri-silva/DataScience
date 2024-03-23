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
    df2 = df_t.rename(columns=constants.COLUNMS_NAME_AGG_TRAGE)
    
    # my_dict = df2.to_dict()
    # #my_dict2 = [my_dict[0]]
    # print("back againnnnnnnnnnnnnnnnnnnnnnnnnnn")
    # df_evt_time = event_time_df(df2['event_time'])
    # df_trd_time = trade_time_df(df2['trade_time'])
    # print(type(my_dict))

    # df2.append(df_evt_time, ignore_index=True)
    # df2.append(df_trd_time, ignore_index=True)
    # my_dict.append(df_evt_time)
    store_data.agg_trade_csv(df2)