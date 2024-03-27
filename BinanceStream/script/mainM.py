from extract import binance_stream_ws
from config import constants


if __name__ == "__main__":
    binance_stream_ws.run_websocket_book_tickek(constants.SYMBOLS_USDT.get("solana"))