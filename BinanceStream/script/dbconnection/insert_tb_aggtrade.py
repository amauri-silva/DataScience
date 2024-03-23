#import postgres_connector


def insert_tb_aggtrade(connection, df):

    data = {
    'aggt_event_type': df.event_type(),
    'aggt_event_time': '2022-01-01 00:00:00',
    'aggt_symbol': 'example_symbol',
    'aggt_aggregate_trade_id': 1,
    'aggt_price': 100.0,
    'aggt_quantity': 2,
    'aggt_first_trade_id': 1,
    'aggt_last_trade_id': 2,
    'aggt_trade_time': '2022-01-01 00:00:00',
    'aggt_is_the_buyer_market_maker': True,
    'aggt_ignore': False,
    'aggt_event_time_dt': '2022-01-01',
    'aggt_event_time_tm': '00:00:00',
    'aggt_trade_time_dt': '2022-01-01',
    'aggt_trade_time_tm': '00:00:00'
}




    query = """
    INSERT INTO tb_aggtrade (
        aggt_event_type, aggt_event_time, aggt_symbol, aggt_aggregate_trade_id, 
        aggt_price, aggt_quantity, aggt_first_trade_id, aggt_last_trade_id, 
        aggt_trade_time, aggt_is_the_buyer_market_maker, aggt_ignore
    ) 
    VALUES (%(aggt_event_type)s, %(aggt_event_time)s, %(aggt_symbol)s, %(aggt_aggregate_trade_id)s, 
            %(aggt_price)s, %(aggt_quantity)s, %(aggt_first_trade_id)s, %(aggt_last_trade_id)s, 
            %(aggt_trade_time)s, %(aggt_is_the_buyer_market_maker)s, %(aggt_ignore)s)
    """
    cursor = connection.cursor()
    try:
        cursor.execute(query, data)
        connection.commit()
        print("Commiting data on tb_aggtrade...")
    except Exception as e:
        print("Error inserting data on tb_aggtrade")
        print(f"The error '{e}' occurred")

'''
def insert_into_table(connection, data):
    query = """
    INSERT INTO tb_aggtrade (
        aggt_event_type, aggt_event_time, aggt_symbol, aggt_aggregate_trade_id, 
        aggt_price, aggt_quantity, aggt_first_trade_id, aggt_last_trade_id, 
        aggt_trade_time, aggt_is_the_buyer_market_maker, aggt_ignore, 
        aggt_event_time_dt, aggt_event_time_tm, aggt_trade_time_dt, aggt_trade_time_tm
    ) 
    VALUES (%(aggt_event_type)s, %(aggt_event_time)s, %(aggt_symbol)s, %(aggt_aggregate_trade_id)s, 
            %(aggt_price)s, %(aggt_quantity)s, %(aggt_first_trade_id)s, %(aggt_last_trade_id)s, 
            %(aggt_trade_time)s, %(aggt_is_the_buyer_market_maker)s, %(aggt_ignore)s, 
            %(aggt_event_time_dt)s, %(aggt_event_time_tm)s, %(aggt_trade_time_dt)s, %(aggt_trade_time_tm)s)
    """
    cursor = connection.cursor()
    try:
        cursor.execute(query, data)
        connection.commit()
        print("Data inserted successfully")
    except Exception as e:
        print(f"The error '{e}' occurred")

'''


# example usage
data = {
    'aggt_event_type': 'example_type',
    'aggt_event_time': '2022-01-01 00:00:00',
    'aggt_symbol': 'example_symbol',
    'aggt_aggregate_trade_id': 1,
    'aggt_price': 100.0,
    'aggt_quantity': 2,
    'aggt_first_trade_id': 1,
    'aggt_last_trade_id': 2,
    'aggt_trade_time': '2022-01-01 00:00:00',
    'aggt_is_the_buyer_market_maker': True,
    'aggt_ignore': False,
    'aggt_event_time_dt': '2022-01-01',
    'aggt_event_time_tm': '00:00:00',
    'aggt_trade_time_dt': '2022-01-01',
    'aggt_trade_time_tm': '00:00:00'
}

# pg_connection  = postgres_connector.postgres_connector()
# insert_into_table(pg_connection, data)