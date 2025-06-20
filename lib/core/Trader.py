import pickle

import lib.core.utils as utils
from lib.core.Action import Action
from lib.core.Position import Position, PositionStatus, PositionType
from lib.core.Strat import Strat
from lib.core.TradeConfig import TradeConfig


class Trader:
    def __init__(self, name, balance, bpf: TradeConfig, verbose=False):
        self.init_balance = balance
        self.name = name
        self.positions_record = []
        self.config = bpf
        self.verbose = verbose
        self.losses = []
        self.gains = []
        self.reset()
    
    def reset(self):
        self.balance = self.init_balance
        self.shares = 0
        self.borrowed_shares = 0
        self.current_position: Position = None
        self.buy_decisions = []
        self.reset_opstate()
    
    def reset_opstate(self):
        self._networth = None
        self._shares_borrowing_cost = None
        self._borrowed_shares_worth = None
        self._borrowing_cost_incurred = None
        self._position_profit = None
        self._held_shares_worth = None
        self._date = None

    def check_cutloss(self, strat: Strat, price):
        if not self.current_position:
            return False
        if strat.cut_loss and self.current_position and self.current_position.type == PositionType.LONG:
            if price < self.current_position.open_price * (1 + strat.long_loss_rpv):
                return True
        
    def check_liquidation(self):
        if not (self.current_position and self.current_position.is_short()):
            return False
        return self.balance < self.config.margin_level * self._borrowing_cost_incurred

    def resfresh_position(self, price, timestamp):
        self._date = utils.paris_datetime(timestamp)
        self._held_shares_worth = self.held_shares_worth(price)
        self._borrowed_shares_worth = self.compute_borrowed_shares_worth(price)
        self._shares_borrowing_cost = self.compute_borrowing_cost(price, timestamp)
        self._borrowing_cost_incurred = self._shares_borrowing_cost + self._borrowed_shares_worth
        self._networth = self.balance + self._held_shares_worth - self._borrowing_cost_incurred
        self._position_profit = 0
        if self.current_position:
            self._position_profit = self._networth - self.current_position.open_balance
            self.current_position.min_profit_reached = min(self._position_profit, self.current_position.min_profit_reached)
            self.current_position.max_profit_reached = max(self._position_profit, self.current_position.max_profit_reached)
                    
    def compute_shares_worth(self, price):
        return self.shares * price * (1 - self.config.tr_fee_rate)

    def compute_position_profit(self, price, timestamp):
        return self.compute_networth(price, timestamp) - self.current_position.open_balance

    def enter_long(self, price, timestamp):
        if self.current_position:
            raise Exception("Already in a position!")        
        self.current_position = Position(
            type=PositionType.LONG,
            open_timestamp=timestamp,
            open_balance=self.balance,
            open_price=price,
            status=PositionStatus.OPEN,
        )
        self.shares = self.balance / ((1 + self.config.tr_fee_rate) * price)
        self.balance = 0
        if self.verbose:
            print(f"[{self._date}] => Enter long at {price:.4f}")

    def enter_short(self, price, timestamp):
        if self.current_position:
            raise Exception("Already in a position!")
        shares_can_borrow = self.balance / (price * self.config.margin_level)
        if shares_can_borrow <= self.config.min_trade_amount:
            raise ValueError("Not enough balance to short.")
        self.current_position = Position(
            type=PositionType.SHORT,
            open_timestamp=timestamp,
            open_price=price,
            open_balance=self.balance,
            status=PositionStatus.OPEN,
        )
        self.balance += shares_can_borrow * price * (1 - self.config.tr_fee_rate)
        self.borrowed_shares = shares_can_borrow
        if self.verbose:
            print(f"[{self._date}] => Enter short at {price}")

    def exit_short(self, price, timestamp, reason="n/a"):
        if not self.current_position.is_short():
            raise ValueError("No short position to exit.")            
        self.balance = self._networth
        self.borrowed_shares = 0
        self.current_position.close(price, timestamp, self.balance, reason)

    def exit_long(self, price, timestamp, reason="n/a"):
        if not self.current_position.is_long():
            raise ValueError("No long position to exit.")
        self.balance = self._networth
        self.shares = 0
        self.current_position.close(price, timestamp, self.balance, reason)

    def close_position(self, price, timestamp, reason="n/a"):
        if not self.current_position:
            raise ValueError("No position to close.")
        if self.verbose:
            print(f"[{self._date}] <= Exit at {price:.4f}, made {self._position_profit:.2f}$, networth: {self._networth:.2f}$")
        if self.current_position.is_short():
            self.exit_short(price, timestamp, reason)
        elif self.current_position.is_long():
            self.exit_long(price, timestamp, reason)
        tmp = self.current_position
        self.positions_record.append(tmp)
        self.current_position = None
        return tmp

    def compute_borrowed_shares_worth(self, price):
        if not self.borrowed_shares:
            return 0
        return self.borrowed_shares * price * (1 + self.config.tr_fee_rate)

    def compute_borrowing_cost(self, price, timestamp):
        if not self.borrowed_shares:
            return 0
        time_held_hours = utils.time_diff_hours(self.current_position.open_timestamp, timestamp)
        return self.borrowed_shares * self.config.hourly_borrow_rate * time_held_hours * price * (1 + self.config.tr_fee_rate)
    
    def held_shares_worth(self, price):
        if not self.shares: 
            return 0
        return self.shares * price * (1 - self.config.tr_fee_rate)
    
    def compute_networth(self, price, timestamp):
        return self.balance + self.held_shares_worth(price) - self.compute_borrowed_shares_worth(price) - self.compute_borrowing_cost(price, timestamp)

    def get_roi(self, price):
        return self.compute_networth(price) / self.init_balance - 1

    def save_state(self, filename=None):
        if filename is None:
            filename = f"{self.name}_trader_state.pickl"
        with open(filename, "w") as f:
            pickle.dump(self, f)

    def load_state(self, filename):
        try:
            with open(filename, 'rb') as file:
                state = pickle.load(file)
                self.__dict__.update(state)
            print(f"State loaded from {filename}")
        except FileNotFoundError:
            print(f"State file '{filename}' not found.")
        except Exception as e:
            print(f"An error occurred while loading the state: {str(e)}")

    def __str__(self):
        return f"{self.name} - Balance: ${self.balance}, Assets: {self.shares}"
    
    def check_take_profit(self, strat: Strat, price):
        if strat.profit_margin and self.current_position and self.current_position.type == PositionType.LONG:
            if self.current_position.open_price * (1 + strat.long_min_rpv) < price:
                return True
            
    def step(self, strat: Strat, price, timestamp, t, verbose, take_shorts):
        self.verbose = verbose
        self.resfresh_position(price, timestamp)
        if self.check_cutloss(strat, price):
            self.losses.append(self._position_profit)
            print(f"[{self._date}] !!! Cut loss at {price:.4f}")
            self.close_position(price, timestamp, f"Cut loss")
        elif self.check_liquidation():
            print(f"[{self._date}] !!! Liquidation at {price:.4f}")
            self.close_position(price, timestamp, f"Liquidation")
        if self.check_take_profit(strat, price):
            self.gains.append(self._position_profit)
            print(f"[{self._date}] !!! Profit margin reached at {price:.4f}")
            self.close_position(price, timestamp, f"Profit margin reached")
        if not self.current_position:
            action = strat.get_action(price, t)
            if action == Action.BUY:
                if not self.current_position:
                    self.enter_long(price, timestamp)
                elif self.current_position.is_short():
                    self.close_position(price, timestamp)
            elif action == Action.SELL:
                if not self.current_position and take_shorts:
                    self.enter_short(price, timestamp)
                elif self.current_position and self.current_position.is_long() and not strat.profit_margin:
                    self.close_position(price, timestamp)
        self.reset_opstate()
 