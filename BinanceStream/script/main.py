from extract import binance_stream_ws
from config import constants


if __name__ == "__main__":
    binance_stream_ws.run_websocket_agg_trade(constants.SYMBOLS_USDT.get("solana"))
    # binance_stream_ws.run_websocket_trade(constants.SYMBOLS_USDT.get("solana"))
    # binance_stream_ws.run_websocket_avg_price(constants.SYMBOLS_USDT.get("solana"))
    # binance_stream_ws.run_websocket_book_tickek(constants.SYMBOLS_USDT.get("solana"))
    # binance_stream_ws.run_websocket_kline_interval(constants.SYMBOLS_USDT.get("solana"))
    # binance_stream_ws.run_websocket_ticker_window(constants.SYMBOLS_USDT.get("solana"))
    #binance_stream_ws.run_websocket_ticker_window_arr(constants.SYMBOLS_USDT.get("solana"))