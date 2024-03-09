
def trade(symbol):
    print()

    asserts = [symbol]
    asserts = [coin.lower() + "@trade" for coin in asserts]
    asserts = '/'.join(asserts)

    return asserts

def agg_trade(symbol):
    """
    The Aggregate Trade Streams push trade information that is aggregated for a single taker order.
    Stream Name: <symbol>@aggTrade
    Update Speed: Real-time
    """
    asserts = symbol.lower() + "@aggTrade"
    return asserts

def kline(symbol, interval):
    asserts = symbol.lower() + "@kline_" + interval
    asserts = '/'.join(asserts)
    return asserts

def mini_ticker(symbol):
    asserts = symbol.lower() + "@miniTicker"
    asserts = '/'.join(asserts)
    return asserts

def ticker_window(symbol, window_size):
    asserts = symbol.lower() + "@ticker_" + window_size
    asserts = '/'.join(asserts)
    return asserts

def ticker_window_arr(window_size):
    asserts = "!ticker_" + window_size + "@arr"
    asserts = '/'.join(asserts)

    return asserts


def book_ticker(symbol):
    asserts = symbol.lower() + "@bookTicker"
    asserts = '/'.join(asserts)

    return asserts


def avg_price(symbol):
    """
    Stream Names: <symbol>@depth<levels> OR <symbol>@depth<levels>@100ms
    Update Speed: 1000ms or 100ms
    """
    asserts = symbol.lower() + "@avgPrice"
    asserts = '/'.join(asserts)

    return asserts

def deth__level(symbol, level):
    asserts = symbol.lower() + "@depth" + level
    asserts = '/'.join(asserts)

    return asserts

def deth__level_100ms(symbol, level):
    """
    Order book price and quantity depth updates used to locally manage an order book.
    Stream Name: <symbol>@depth OR <symbol>@depth@100ms
    Update Speed: 1000ms or 100ms
    """
    asserts = symbol.lower() + "@depth" + level + "@100ms"
    asserts = '/'.join(asserts)

    return asserts