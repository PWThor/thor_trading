from ib_insync import IB, Stock, Future, Order
import logging
import re
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename=r'E:\Projects\thor_trading\outputs\logs\ib_trading.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_signals(log_file):
    """Load trade signals from the ml_trading_signals.log file."""
    signals = []
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Signal for" in line and "Hold" not in line:
                # Example log line: "Signal for CL: Buy at 63.6100, Take-Profit: 65.5183, Stop-Loss: 62.9200, Contracts: 1, Rationale: ML Prediction (Probability Up: 0.72)"
                symbol_match = re.search(r"Signal for (\w+):", line)
                signal_type_match = re.search(r": (\w+) at", line)
                entry_price_match = re.search(r"at ([\d.]+),", line)
                take_profit_match = re.search(r"Take-Profit: ([\d.]+),", line)
                stop_loss_match = re.search(r"Stop-Loss: ([\d.]+),", line)
                contracts_match = re.search(r"Contracts: (\d+),", line)
                prob_match = re.search(r"Probability Up: ([\d.]+)", line)

                if all([symbol_match, signal_type_match, entry_price_match, take_profit_match, stop_loss_match, contracts_match, prob_match]):
                    signal = {
                        'symbol': symbol_match.group(1),
                        'signal_type': signal_type_match.group(1),
                        'entry_price': float(entry_price_match.group(1)),
                        'take_profit': float(take_profit_match.group(1)),
                        'stop_loss': float(stop_loss_match.group(1)),
                        'contracts': int(contracts_match.group(1)),
                        'probability': float(prob_match.group(1))
                    }
                    signals.append(signal)
    return signals

def connect_to_ib_gateway():
    """Connect to IB Gateway."""
    ib = IB()
    try:
        ib.connect('127.0.0.1', 4001, clientId=1)  # Port 4001 for paper trading
        logging.info("Connected to IB Gateway successfully.")
        return ib
    except Exception as e:
        logging.error(f"Failed to connect to IB Gateway: {str(e)}")
        raise

def place_trade(ib, signal):
    """Place a trade using IB Gateway based on the signal."""
    symbol = signal['symbol']
    signal_type = signal['signal_type']
    entry_price = signal['entry_price']
    take_profit = signal['take_profit']
    stop_loss = signal['stop_loss']
    contracts = signal['contracts']

    # Define the contract (CL and HO are futures on NYMEX)
    contract = Future(symbol=symbol, exchange='NYMEX', currency='USD', lastTradeDateOrContractMonth='202506')

    # Qualify the contract to ensure it’s valid
    ib.qualifyContracts(contract)
    logging.info(f"Qualified contract: {contract}")

    # Define the main order (limit order at entry price)
    if signal_type == "Buy":
        action = "BUY"
        limit_price = entry_price
    else:  # Sell
        action = "SELL"
        limit_price = entry_price

    main_order = Order(
        action=action,
        totalQuantity=contracts,
        orderType="LMT",
        lmtPrice=limit_price,
        transmit=False  # Don’t transmit yet; we’ll attach bracket orders
    )

    # Define take-profit order
    take_profit_order = Order(
        action="SELL" if action == "BUY" else "BUY",
        totalQuantity=contracts,
        orderType="LMT",
        lmtPrice=take_profit,
        transmit=False,
        parentId=main_order.orderId
    )

    # Define stop-loss order
    stop_loss_order = Order(
        action="SELL" if action == "BUY" else "BUY",
        totalQuantity=contracts,
        orderType="STP",
        auxPrice=stop_loss,
        transmit=True,  # Transmit the entire bracket
        parentId=main_order.orderId
    )

    # Place the bracket order
    bracket = [main_order, take_profit_order, stop_loss_order]
    for order in bracket:
        trade = ib.placeOrder(contract, order)
        logging.info(f"Placed order: {order.action} {order.totalQuantity} {symbol} at {order.lmtPrice or order.auxPrice}")
        ib.sleep(1)  # Brief pause to avoid pacing violations

    return trade

def main():
    logging.info("Starting IB Gateway trading execution script...")

    # Load signals from the log file
    log_file = r'E:\Projects\thor_trading\outputs\logs\ml_trading_signals.log'
    signals = load_signals(log_file)
    if not signals:
        logging.warning("No trade signals found in the log file.")
        print("No trade signals found to execute.")
        return

    # Connect to IB Gateway
    ib = connect_to_ib_gateway()

    # Place trades for each signal
    for signal in signals:
        try:
            trade = place_trade(ib, signal)
            print(f"Executed trade for {signal['symbol']}: {signal['signal_type']} at {signal['entry_price']:.4f}")
        except Exception as e:
            logging.error(f"Failed to place trade for {signal['symbol']}: {str(e)}")
            print(f"Failed to place trade for {signal['symbol']}: {str(e)}")

    # Disconnect from IB Gateway
    ib.disconnect()
    logging.info("Disconnected from IB Gateway.")

if __name__ == "__main__":
    main()