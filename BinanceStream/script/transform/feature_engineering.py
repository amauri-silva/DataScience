import pandas as pd
from config import constants
from load import store_data

def transform_agg_trade(json):
    df = pd.DataFrame.from_dict(json['data'], orient='index')
    df_t = df.transpose()
    df2 = df_t.rename(columns=constants.COLUNMS_NAME_AGG_TRAGE)
    store_data.csv(df2)