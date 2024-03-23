import pandas as pd
from dbconnection import insert_tb_aggtrade, postgres_connector

def agg_trade_csv(data_frame):
    df1 = data_frame
    #df1.to_csv('agg_trade_solusdt.csv', mode='a', header=False, index=False, sep=',')
    
    # Save to CSV and run a bash process to load the original data on database
    print("Opening connection to PostgreSQL DB...")
    pg_connection  = postgres_connector.postgres_connector()
    print("##########################:   {}".format(df1['event_time']))
    insert_tb_aggtrade.insert_tb_aggtrade(pg_connection, df1)
    print("Data saved successfully on tb_agg_trade")
    