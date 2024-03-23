AGGTRADE = "@aggTrade"
TRADE = "@trade" 
KLINE_= "@kline_"
TIME_INTERVAL = {"1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}
MINITICKER = "@miniTicker"
TICKER = "@ticker"
TICKER_ = "@ticker_" 
ARR = "@arr" 
BOOKTICKER = "@bookTicker" 
AVGPRICE = "@avgPrice"
DEPTH = "@depth"
NUMBER_INTERVAL = {"5":"5", "10":"10", "20":"20", "@100ms":"@100ms"}
SYMBOLS_USDT = {"solana":"SOLUSDT"}


#COLUNMS_NAME_AGG_TRAGE = {"e":"event_type", "E":"event_time", "s":"symbol", "a":"aggregate_trade_id", "p":"price", "q":"quantity", "f":"first_trade_id", "l":"last_trade_id", "T":"trade_time", "m":"is_the_buyer_market_maker", "M":"ignore"}
COLUNMS_NAME_AGG_TRAGE = {"e":"aggt_event_type", "E":"aggt_event_time", "s":"aggt_symbol", "a":"aggt_aggregate_trade_id", "p":"aggt_price", "q":"aggt_quantity", "f":"aggt_first_trade_id", "l":"aggt_last_trade_id", "T":"aggt_trade_time", "m":"aggt_is_the_buyer_market_maker", "M":"aggt_ignore"}