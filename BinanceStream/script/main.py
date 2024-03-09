from extract import agg_trade_getstream
from config import constants


if __name__ == "__main__":

    agg_trade_getstream.run_websocket_agg_trade(constants.SYMBOLS_USDT.get("solana"))