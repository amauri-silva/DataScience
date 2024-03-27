
# def trade(symbol):
#     print()

#     asserts = [symbol]
#     asserts = [coin.lower() + "@trade" for coin in asserts]
#     asserts = '/'.join(asserts)

#     return asserts

def agg_trade(symbol):
    """
    The Aggregate Trade Streams push trade information that is aggregated for a single taker order.
    Stream Name: <symbol>@aggTrade
    Update Speed: Real-time
    """
    asserts = symbol.lower() + "@aggTrade"
    return asserts

def trade(symbol):
    """
    The Trade Streams push raw trade information; each trade has a unique buyer and seller.
    Stream Name: <symbol>@trade
    """
    asserts = symbol.lower() + "@trade"
    return asserts

def avg_price(symbol):
    """
    Average price streams push changes in the average price over a fixed time interval.
    Stream Name: <symbol>@avgPrice
    Update Speed: 1000ms
    """
    asserts = symbol.lower() + "@avgPrice"
    return asserts

def kline(symbol, interval):
    """
    The Kline/Candlestick Stream push updates to the current klines/candlestick every second.
    Kline/Candlestick chart intervals:
    s-> seconds; m -> minutes; h -> hours; d -> days; w -> weeks; M -> months
    1s,1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M
    Stream Name: <symbol>@kline_<interval>
    Update Speed: 1000ms for 1s, 2000ms for the other intervals
    """
    asserts = symbol.lower() + "@kline_" + interval
    return asserts

def mini_ticker(symbol):
    asserts = symbol.lower() + "@miniTicker"
    return asserts

def ticker_window(symbol, window_size):
    asserts = symbol.lower() + "@ticker_" + window_size
    return asserts

def ticker_window_arr(window_size):
    asserts = "!ticker_" + window_size + "@arr"
    return asserts


def book_ticker(symbol):
    """
    Pushes any update to the best bid or ask's price or quantity in real-time for a specified symbol. Multiple <symbol>@bookTicker 
    streams can be subscribed to over one connection.
    Stream Name: <symbol>@bookTicker
    Update Speed: Real-time
    """
    asserts = symbol.lower() + "@bookTicker"
    return asserts



def deth__level(symbol, level):
    asserts = symbol.lower() + "@depth" + level
    return asserts

def deth__level_100ms(symbol, level):
    """
    Order book price and quantity depth updates used to locally manage an order book.
    Stream Name: <symbol>@depth OR <symbol>@depth@100ms
    Update Speed: 1000ms or 100ms
    """
    asserts = symbol.lower() + "@depth" + level + "@100ms"
    return asserts