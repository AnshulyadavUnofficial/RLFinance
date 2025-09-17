import neat
from typing import NamedTuple, Optional, Dict, List
from enum import Enum
import numpy as np
from tf_agents.environments import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import random
from pathlib import Path
import pandas as pd
from pprint import pprint
import _cppneat
import neat.activations as neat_act
import neat.aggregations as neat_agg# where sigmoid_activation etc live


class DataSlice(NamedTuple):
    time: float = None
    open: float = None
    high: float = None
    low: float = None
    close : float = None
    volume : int = None
    historical_avg_vol : int = None    

  
def cache_directory_files(directory: str, exclude_sub_dir: Optional[str] = None, num_prev_file=0):
    """
    Collects and caches all CSV files in each immediate subdirectory of `directory`,
    converting them to normalized arrays.
    
    Args:
        directory: Path of the top-level directory whose subdirectories contain CSVs
        exclude_sub_dir: Name of a subdirectory to fully exclude (optional)
    
    Returns:
        Dictionary mapping relative file paths (from `directory`) to DataSlice objects.
    
    Raises:
        FileNotFoundError: If no valid files are found
    """
    root = Path(directory)
    cache = {}
    new_cache = {}
    directory_group:list[Path] = []
    
    if exclude_sub_dir is not None: exclude_sub_dir = exclude_sub_dir.split("/")[-1]
    
    # either the root is s directory of directories, each of which contains files
    # or a directory that contains files.
    for sub_dir in root.iterdir():
        # root is a directory of files
        if sub_dir.is_file(): 
            directory_group.append(root)
            break
        # root is a directory of directories
        if exclude_sub_dir != sub_dir.name:
            directory_group.append(sub_dir)
    
    # process the groups, each directory in group, is presumed to have only files and a date_file.txt
    for directory in directory_group:
        # Iterate files in this subdirectory
        for path in directory.iterdir():
            if not path.is_file():continue
            if "date_file.txt" == path.name:continue

            # Read CSV: assume 'Time' index column exists
            df = pd.read_csv(path, header=0, index_col="Time")

            # Convert time index to minutes since 09:30
            # e.g., if index is "09:45", "09:45:00" minus 09:30
            time_deltas = pd.to_timedelta(df.index + ":00") - pd.Timedelta(hours=9, minutes=30)
            times = (time_deltas.total_seconds() / 60).astype(np.float32).values

            # Extract OHLCV columns as float32 arrays
            bars = df[["Open", "High", "Low", "Close", "Volume"]].to_numpy(np.float32).T
            
            # Store with relative path from root, e.g. "subdir/filename.csv"
            cache[str(path)] = DataSlice(
                time=times,
                open=bars[0],
                high=bars[1],
                low=bars[2],
                close=bars[3],
                volume=bars[4]
            )
            
        if num_prev_file == 0: continue
        # a valid directory, read the date file in order.
        file_order = []
        done_files = set()
        
        with open(directory.joinpath("date_file.txt"),"r") as file:
            for (ind,line) in enumerate(file):
                file_path = directory.joinpath(line.strip())
                file_order.append(str(file_path))
        
        # write the historical average for volume
        for (ind, file) in enumerate(file_order):
            if ind <num_prev_file:continue
            if file in done_files: continue # dont repeat the file
            
            rel_volume = np.zeros_like(cache[file].volume)
            
            for i in range(ind-1, ind-(num_prev_file+1), -1): # because stop in range is exclusive
                file_name = file_order[i]
                rel_volume += cache[file_name].volume
            rel_volume  = rel_volume/num_prev_file
            
            curr_slice = cache[file]
            new_cache[file]= DataSlice(
                time=curr_slice.time,
                open= curr_slice.open,
                close=curr_slice.close,
                low = curr_slice.low,
                high=curr_slice.high,
                volume=curr_slice.volume,
                historical_avg_vol=np.copy(rel_volume)
            )
            
            done_files.add(file)
    return cache if num_prev_file ==0 else new_cache



class SimpleState(Enum):
    neutral = 0
    short = 1
    long = 2
    hold = 3
    
class USStockActionEnvSimple(PyEnvironment):
    
    def __init__(self, data_cache: Dict[str, DataSlice], num_look_back_period = 10, return_literal_money:bool = False):
        """
        Initialize the trading environment.

        Args:
            data_cache (Dict[str, DataSlice]):
                    - for each file include, the file_path from its directory root, to DataSlice of OHLCV and timedeltas in minutes
            return_literal_money (bool):
                - False (default): “training” mode. Rewards are log-returns
                (log(P_sell / P_buy)), and data files are sampled at random
                to diversify training.
                - True: “evaluation” mode. Rewards are literal dollar profits
                (P_sell - P_buy), and a fixed ticker directory is used so you
                can measure real-world P&L.
        Observation: (3n+6) -dimensional float vector (all scaled to ~10⁻¹-10¹) consisting of:
            1,2,3 One-hot of previous action states 
            4. Minutes since market open
            5. Minutes since last actionable 
            6. Trend strength and direction [-1,1] sum(dlogprice)/sum(abs(dlongprice)) -> +1 long trend, -> -1 short trend, 0-> vacillating.
            n Log price change (log(P_close / P_open))
            n Log volume change (log(curr_vol / historical_avg_vol))
            n close to mid point changes (close - (high + low)/2)/(high - low + epsilon)/2
        """
        self._n = num_look_back_period
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=( (3*self._n + 6),), dtype=np.float32, name='observation')
        
        # Some Variables
        self._end_action_time_delta = 389.0
        self._return_literal_money = return_literal_money
        self._data_cache = data_cache
        self._files = list(self._data_cache.keys())
        self._FILE_NAME = None # current file for logging purposes
        
        self._one_hot_map = {
            SimpleState.neutral: np.array([1,0,0], dtype=np.float32),
            SimpleState.short: np.array([0,1,0], dtype=np.float32),
            SimpleState.long: np.array([ 0,0,1], dtype=np.float32),
        }
        
        # observation scale factors
        self._time_scale_factor =                10**(-2)
        self._stock_change_scale_factor =         10**3
        self._volume_scale_factor =              1 # dlogvol are around 0.4, so keep then as is
        self._close_to_mid_ratio_scale_factor =  1
        
        
        """ Rewards """
        # illegal action reward when the agent takes an action not allowed by the action cycle like short to long or long to short
        self._illegal_action_penalty =   -5 / 1000
       
        # a single side fee rate that is applied individually on both buy and sell
        self._per_side_fee_rate =        0.0003
        
        
        """ logging purposes """
        self.logger: Optional[List[Dict]] = None
        self.logging_step: Optional[int] = None
        
        self._reset()

    def action_spec(self): return self._action_spec
    
    def observation_spec(self): return self._observation_spec
    
    def _reset(self):
        # Following vars form the observation for the agent
        self._market_start_time_delta = None            # time elapased since the market start time
        self._prev_action_time_delta = None             # time since last non-hold action (short, long, neutral)

        self._stock_changes = None                          # a n element list of log(close/open) for n time periods                     
        self._volume_changes = None                         # a n element list of log(curr_volume/histoircal avg volume) for n time periods
        self._close_to_mid_ratio = None                     # a n element list of (c - (h+l)/2)/(h-l)
        
        # previous non-hold action (short, long, neutral) If the current is not previous action then this variable is the current action
        self._prev_action = None                 
        
        
        # Internal States
        self._ranges = None
        self._prev_price = None # set in short/long. Is in $, ie not normalized.
        
        # Clear previous data and initialize new data in ranges
        self._FILE_NAME = random.choice(self._files)
        self._file_df = self._data_cache[self._FILE_NAME]
        self._start_ind = 0
        self._ranges = self._get_short_slice(self._file_df, self._start_ind, self._n)
       
        # Set previous stock value and time based on last available stocks and times
        self._market_start_time_delta = self._ranges.time[-1]   # n minutes since market start
        self._prev_action_time_delta = self._ranges.time[-1]    # n minutes since the neutral action
        
        # calulate price change in the lastest range
        # new values towards the end of the list, and old values towards the front of the list
        self._stock_changes = np.log(self._ranges.close) - np.log(self._ranges.open)
    
        # Volume
        self._volume_changes = np.log(self._ranges.volume) -np.log(self._ranges.historical_avg_vol)
        
        # close to mid point
        num = 2*(self._ranges.close - (self._ranges.high + self._ranges.low)/2)
        den = self._ranges.high - self._ranges.low
        ratio = np.zeros_like(num, dtype=np.float32)
        self._close_to_mid_ratio =np.divide(num, den, out=ratio, where=(den != 0))

        # Define the initial state, assuming it adheres to the observation spec
        self._prev_action = SimpleState.neutral # neutral action taken at time t=n
        
        self._state = self._make_state()
        
        # Prepration for step function
        self._is_episode_over = False
        self._start_ind += 1
            
        # Return the initial time step with the state
        return ts.restart(self._state)

    def _step(self, action):
        if self._is_episode_over : return self._reset()
        action = SimpleState(int(action))
        self._ranges = self._get_short_slice(self._file_df, self._start_ind, self._n)
        
        """ Update metrics based on new data """
        # calculate new start time delta and prev time delta
        # the difference between previous start prediction time delta and current latest time is an interval
        # so add an interval preemptively and reset to 0, for an actual action
        self._prev_action_time_delta += self._ranges.time[-1] - self._market_start_time_delta
        self._market_start_time_delta = self._ranges.time[-1]
        
        # stock and volume changes
        self._stock_changes = np.log(self._ranges.close) - np.log(self._ranges.open)
        # Volume
        self._volume_changes = np.log(self._ranges.volume) -np.log(self._ranges.historical_avg_vol)
        # close to mid point
        num = 2*(self._ranges.close - (self._ranges.high + self._ranges.low)/2)
        den = self._ranges.high - self._ranges.low
        ratio = np.zeros_like(num, dtype=np.float32)
        self._close_to_mid_ratio =np.divide(num, den, out=ratio, where=(den != 0))
        
        # new stock price
        new_stock_price = self._ranges.close[-1]
        
        # EOD handling
        if self._market_start_time_delta >= self._end_action_time_delta:
            raw = 0.0 # preassign raw reward
            
            # settle any open positions, take agent's prediction into account
            if self._prev_action in {SimpleState.long, SimpleState.short}:
                if self._return_literal_money:
                    if self._prev_action == SimpleState.short:
                        # forcing a buy, buys are negative
                        reward = -new_stock_price * (1 + self._per_side_fee_rate)
                    else:
                        # forcing a sell, sells are positive 
                        reward = +new_stock_price * (1 - self._per_side_fee_rate)
                else:
                    if action == SimpleState.neutral: # if the agent correctly predicted
                        if self._prev_action == SimpleState.long:
                            sell_price = new_stock_price
                            buy_price = self._prev_price  # Use actual entry price
                        elif self._prev_action == SimpleState.short:
                            buy_price = new_stock_price
                            sell_price = self._prev_price
                        
                        raw = np.log( (sell_price * (1 - self._per_side_fee_rate) ) / (buy_price * ( 1 + self._per_side_fee_rate)) )
                        reward = raw
                    
                    else: reward = self._illegal_action_penalty   
                # Force a neutral action
                self._prev_action_time_delta = 0
                self._prev_action = SimpleState.neutral
                
            else: # prev action is neutral
                assert self._prev_action == SimpleState.neutral
                if action in {SimpleState.short, SimpleState.long}:
                    if self._return_literal_money: reward = 0 # no reward since no action will be taken
                    else: reward = self._illegal_action_penalty # penalize if the agent is trying to open a position
                else:  
                    if self._return_literal_money: reward = 0 
                    else: # no missed move penalty, since no action can be taken after eod. 
                        # If the agent completed one timestep earlier, then no penalty is applied, 
                        # otherwise the agent must have been penalized enough for holding in the normal branches.
                        reward = 0
                    
            # build final state and terminate
            self._is_episode_over = True
            self._state = self._make_state()
    
            # log states before reward scaling
            if self.logger is not None and self.logging_step is not None:
                self.logger.append({
                    'step':  self.logging_step,
                    'raw':   raw,
                    'reward': reward,
                    'bonus': reward - raw,
                    'is_legal': 1,
                    'action': self._prev_action.name,
                    'data': new_stock_price,
                })
        
            return ts.termination(self._state, reward)

        """
        Action Cycle Transitions
        """
        # Assume that the current action is legal for downstream reward assignment
        is_legal_action = True
        
        # Handle short action
        if action == SimpleState.short:
            # If the previous action was neutral, initiate a short cycle
            if self._prev_action == SimpleState.neutral:
                self._prev_action_time_delta = 0
                
            else: # if previous action was short or long. Previous action cant be hold
                is_legal_action = False
                
        # Handle long action
        elif action == SimpleState.long:
            # If the previous action was neutral start a long cycle
            if self._prev_action == SimpleState.neutral:
                self._prev_action_time_delta = 0
                
            else: # if previous action was short or long. Previous action cant be hold
                is_legal_action = False
                
        # Handle neutral action
        elif action == SimpleState.neutral:
            # if in an active cycle then return back to neutral
            if self._prev_action in {SimpleState.long, SimpleState.short}:
                self._prev_action_time_delta = 0
            
            else: # if previous action was neutral. Previous action cant be hold
                is_legal_action = False
        
        elif action == SimpleState.hold: pass # do nothing in case of hold
            
        else: raise ValueError(f"Encountered unknown action: {action}")
        
        """
        Reward Calculation: calculation post state updates. All new actions are legal but not all legal actions are new.
        If is_new_action, then action cannot be hold
        """
        raw = 0.0       # only assign in normal case reward
        reward = 0.0
        is_new_action = self._prev_action_time_delta==0
    
        # a hold action could either be legal or an illegal override
        if not is_legal_action: action = SimpleState.hold 
        if is_new_action: assert action != SimpleState.hold
        if not is_new_action: assert action == SimpleState.hold
        
        """ Reward Calculation and Fee Assignment: Reward Assigned on new action"""
        # all new actions by default legal
        if is_new_action:
            if self._return_literal_money:
                # starting a shortsell cycling or completing a longsell
                if (action == SimpleState.short) or (action == SimpleState.neutral and self._prev_action == SimpleState.long):
                    reward = +new_stock_price * (1 - self._per_side_fee_rate)
                # starting a longbuy cycle or completing a shortbuy
                elif (action == SimpleState.long) or (action == SimpleState.neutral and self._prev_action == SimpleState.short):
                    reward = -new_stock_price * (1 + self._per_side_fee_rate)
            else: #normal case reward
                if action in {SimpleState.short, SimpleState.long}:
                    self._prev_price = new_stock_price # assign for p & L computation
                
                elif action == SimpleState.neutral:
                    # Long: currently selling, Short: prev price selling
                    sell_price = new_stock_price if self._prev_action == SimpleState.long else self._prev_price
                    # Long: prev price selling, Short: currently buying
                    buy_price = new_stock_price if self._prev_action == SimpleState.short else self._prev_price
                    
                    raw = np.log( (sell_price * (1 - self._per_side_fee_rate) ) / (buy_price * ( 1 + self._per_side_fee_rate)) )
                    reward = raw
                    
                    self._prev_price = None # sanity check to prevent code creeps
                
                else: raise ValueError("Action must be short/long/neutral in the direct trade reward")
            
        elif not is_legal_action:
            reward = self._illegal_action_penalty
            
        # Update state for regular transitions
        if is_new_action: self._prev_action = action # assign prev action in case of changes
        assert self._prev_action != SimpleState.hold
        
        self._state = self._make_state()
        self._start_ind += 1
        
        # log states before reward scaling
        if self.logger is not None and self.logging_step is not None:
            self.logger.append({
                'step':  self.logging_step,
                'raw':   raw,
                'reward': reward,
                'bonus': reward - raw,
                'is_legal': int(is_legal_action),
                'action': self._prev_action.name,
                'data': new_stock_price,
            })
            
        return ts.transition(self._state, reward)

    def _make_state(self):
        """
         Observation: (3n+6) -dimensional float vector (all scaled to ~10⁻¹-10¹) consisting of:
            1,2,3 One-hot of previous action states 
            4. Minutes since market open
            5. Minutes since last actionable 
            6. Trend strength and direction [-1,1] sum(dlogprice)/sum(abs(dlongprice)) -> +1 long trend, -> -1 short trend, 0-> vacillating.
            n Log price change (log(P_close / P_open))
            n Log volume change (log(curr_vol / historical_avg_vol))
            n close to mid point changes (close - (high + low)/2)/(high - low + epsilon)/2
        """
        one_hot_vec = self._one_hot_map[self._prev_action]
        time_stuff = np.array([self._market_start_time_delta,self._prev_action_time_delta], dtype=np.float32)
        time_stuff = time_stuff*self._time_scale_factor
       
        trend_direction = np.array([ np.sum(self._stock_changes)/np.sum(np.abs(self._stock_changes)) ], dtype=np.float32)
        
        stock_changes = self._stock_changes*self._stock_change_scale_factor
        volume_changes = self._volume_changes*self._volume_scale_factor
        closet_to_mid_ratio = self._close_to_mid_ratio*self._close_to_mid_ratio_scale_factor
        
        return np.concatenate([
            one_hot_vec,
            time_stuff,
            trend_direction,
            stock_changes,
            volume_changes,
            closet_to_mid_ratio,
        ], axis=0, dtype=np.float32)
        
    def _get_short_slice(self,file_data: DataSlice, start_ind:int, length_slices:int):
        """
        Extract a contiguous window from a full‐length DataSlice.

        Args:
            file_data:       A DataSlice whose fields are full‐length NumPy arrays.
            start_ind:       The index at which to begin the slice.
            length_slices:   The number of consecutive entries to include.

        Returns:
            A new DataSlice containing views of each field
            from file_data[start_ind : start_ind + length_slices].
        """
        return DataSlice(
            time = file_data.time[start_ind: start_ind + length_slices],
            close = file_data.close[start_ind: start_ind + length_slices],
            open = file_data.open[start_ind: start_ind + length_slices],
            high = file_data.high[start_ind: start_ind + length_slices],
            low = file_data.low[start_ind: start_ind + length_slices],
            volume = file_data.volume[start_ind: start_ind + length_slices],
            historical_avg_vol= file_data.historical_avg_vol[start_ind: start_ind + length_slices],
        )
     

class TradingEvaluator:
    def __init__(self, data_cache: Dict[str, DataSlice], num_look_back_period: int = 10):
        self._data_cache = data_cache
        self._n = num_look_back_period
        self._files = list(data_cache.keys())
        
        # Reward calculation parameters
        self._illegal_action_penalty = -5 / 1000
        self._per_side_fee_rate = 0.0003
        self._end_action_time_delta = 389.0
        
        # Observation scaling factors
        self._time_scale_factor = 1e-2
        self._stock_change_scale_factor = 1e3
        self._volume_scale_factor = 1
        self._close_to_mid_ratio_scale_factor = 1
        
        # Sharpe ratio parameters
        per_year_bond_yield = 5
        trading_days = 250.0
        self._risk_free_rate = (1 + per_year_bond_yield/100)**(1/trading_days) - 1
        self._epsilon = 1e-6
        
        # One-hot encoding for actions
        self._one_hot_map = {
            SimpleState.neutral: np.array([1, 0, 0], dtype=np.float32),
            SimpleState.short: np.array([0, 1, 0], dtype=np.float32),
            SimpleState.long: np.array([0, 0, 1], dtype=np.float32),
        }

    def reset(self):
        """Initialize environment state for a new episode."""
        self._FILE_NAME = np.random.choice(self._files)
        self._file_df = self._data_cache[self._FILE_NAME]
        self._start_ind = 0
        
        # Initialize time tracking
        self._market_start_time_delta = 0.0
        self._prev_action_time_delta = 0.0
        
        # Initialize price history buffers
        self._stock_changes = np.zeros(self._n, dtype=np.float32)
        self._volume_changes = np.zeros(self._n, dtype=np.float32)
        self._close_to_mid_ratio = np.zeros(self._n, dtype=np.float32)
        
        # Initialize position state
        self._prev_action = SimpleState.neutral
        self._prev_price = None
        self._is_episode_over = False
        
        # Load initial data window
        self._load_data_slice()
        # This cannot be assigned via load data slice function, so assign explicitly
        self._prev_action_time_delta = self._market_start_time_delta 
        # prepare for step function
        self._start_ind += 1
        
        return self._make_state()

    def _load_data_slice(self):
        """Load the current data window and update metrics."""
        start, end = self._start_ind, self._start_ind + self._n
        self._market_start_time_delta = self._file_df.time[end-1]
        
        closes = self._file_df.close[start:end]
        # Calculate price changes
        self._stock_changes = np.log(closes) - np.log(self._file_df.open[start:end])
        
        # Calculate volume changes
        self._volume_changes = np.log(self._file_df.volume[start:end]) - np.log(self._file_df.historical_avg_vol[start:end])
        
        # Calculate close-to-mid ratios
        highs = self._file_df.high[start:end]
        lows = self._file_df.low[start:end]
        mids = (highs + lows) / 2
        ranges = highs - lows
        np.divide(2 * (closes - mids), ranges, out=self._close_to_mid_ratio, where= (ranges != 0))

    def _make_state(self):
        """
         Observation: (3n+6) -dimensional float vector (all scaled to ~10⁻¹-10¹) consisting of:
            1,2,3 One-hot of previous action states 
            4. Minutes since market open
            5. Minutes since last actionable 
            6. Trend strength and direction [-1,1] sum(dlogprice)/sum(abs(dlongprice)) -> +1 long trend, -> -1 short trend, 0-> vacillating.
            n Log price change (log(P_close / P_open))
            n Log volume change (log(curr_vol / historical_avg_vol))
            n close to mid point changes (close - (high + low)/2)/(high - low + epsilon)/2
        """
        # One-hot encode previous action
        one_hot = self._one_hot_map[self._prev_action]
        
        # Time features
        time_features = np.array([
            self._market_start_time_delta,
            self._prev_action_time_delta
        ], dtype=np.float32) * self._time_scale_factor
        
        # Trend strength
        trend = np.array([np.sum(self._stock_changes) / np.sum(np.abs(self._stock_changes))], dtype=np.float32)
        
        # Apply scaling to features
        scaled_stock = self._stock_changes * self._stock_change_scale_factor
        scaled_volume = self._volume_changes * self._volume_scale_factor
        scaled_mid_ratio = self._close_to_mid_ratio * self._close_to_mid_ratio_scale_factor
        
        return np.concatenate([
            one_hot,
            time_features,
            trend,
            scaled_stock,
            scaled_volume,
            scaled_mid_ratio
        ])

    def step(self, action: int):
        if self._is_episode_over:
            return 0.0, True
            
        # Convert action to enum type
        action = SimpleState(action)
        
        # Load new data window
        self._load_data_slice()
        self._prev_action_time_delta += 1.0 
        
        # Get current price
        current_price = self._file_df.close[self._start_ind + self._n - 1]
        
        # End-of-day handling
        if self._market_start_time_delta >= self._end_action_time_delta:
            reward = 0.0
            
            # Settle open positions if any
            if self._prev_action in {SimpleState.long, SimpleState.short}:
                if action == SimpleState.neutral:
                    # Calculate P&L for closing position
                    if self._prev_action == SimpleState.long:
                        sell_price = current_price
                        buy_price = self._prev_price
                    else:  # short
                        buy_price = current_price
                        sell_price = self._prev_price
                    
                    reward = np.log( sell_price * (1 - self._per_side_fee_rate) / (buy_price * (1 + self._per_side_fee_rate)))
                else:
                    # Penalize for not closing position at EOD
                    reward = self._illegal_action_penalty
                
                # Reset to neutral position
                self._prev_action = SimpleState.neutral
                self._prev_price = None
                self._prev_action_time_delta = 0.0
            else:  # Already in neutral position
                if action in {SimpleState.short, SimpleState.long}:
                    # Penalize for trying to open position at EOD
                    reward = self._illegal_action_penalty
            
            self._is_episode_over = True
            return self._make_state(), reward, True
        
        """ Action Cycling: Assume that the current action is legal but not new. """
        is_legal_action = True
        is_new_action = False
        
        # Handle action cycle transitions
        if action == SimpleState.short:
            if self._prev_action == SimpleState.neutral:
                is_new_action = True
                self._prev_action_time_delta = 0.0
            else:
                is_legal_action = False
                
        elif action == SimpleState.long:
            if self._prev_action == SimpleState.neutral:
                is_new_action = True
                self._prev_action_time_delta = 0.0
            else:
                is_legal_action = False
                
        elif action == SimpleState.neutral:
            if self._prev_action in {SimpleState.long, SimpleState.short}:
                is_new_action = True
                self._prev_action_time_delta = 0.0
            else:
                is_legal_action = False
                
        # Hold action is always legal, but never new
        
        """ Reward Calculation: All new actions are legal by default, but not all legal actions are new, ie holds.
                                Hold is a legal but not new action."""
        reward = 0.0
        
        # Handle illegal actions
        if not is_legal_action:
            # Override to hold and penalize
            action = SimpleState.hold
            reward = self._illegal_action_penalty
        elif is_new_action:
            if action in {SimpleState.short, SimpleState.long}:
                # Record entry price for future P&L calculation
                self._prev_price = current_price
                
            elif action == SimpleState.neutral:
                # Calculate P&L for closing position
                if self._prev_action == SimpleState.long:
                    sell_price = current_price
                    buy_price = self._prev_price
                else:  # short
                    buy_price = current_price
                    sell_price = self._prev_price
                
                reward = np.log( sell_price * (1 - self._per_side_fee_rate) / (buy_price * (1 + self._per_side_fee_rate)) )
                
                # sanity check: reset prev check so that an error would be thrown
                self._prev_price = None
            else:
                raise ValueError("Invalid action type for new action")
        
        # Update state for new actions (except closing which is handled above)
        if is_new_action: # holds are not new actions
            self._prev_action = action
        
        state = self._make_state()
        # Prepare for next step
        self._start_ind += 1
        
        return state, reward, False

    def __call__(self, genome, config):
        """Evaluate genome's trading performance using Sharpe ratio."""
        state = self.reset()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        returns = []
        
        while not self._is_episode_over:
            action = np.argmax(net.activate(state))
            state, reward, done = self.step(action)
            returns.append(reward)
            if done:
                break
        if len(returns) == 0:
            return -np.inf   # in case the agent took no action
    
        # Calculate Sharpe ratio
        returns = np.array(returns)
        excess_returns = returns - self._risk_free_rate
        sharpe = np.mean(excess_returns) / (np.std(excess_returns) + self._epsilon)
        return sharpe




class ActFn(Enum):
    SIGMOID = 0
    RELU    = 1
    TANH    = 2

class AggFn(Enum):
    SUM  = 0
    MAX  = 1
    MEAN = 2

def build_cppnet(input_nodes, output_nodes, node_evals):
    """
    Ensures the Python side functions are the same objects
    that CPPNet looks for, then constructs the C++ network.
    node_evals: list of (node, act_fn, agg_fn, bias, resp, links)
    """
    # verify callables match
    for _, act, agg, *_ in node_evals:
        assert act in (
            neat_act.sigmoid_activation,
            neat_act.relu_activation,
            neat_act.tanh_activation
        ), "Unknown activation"
        assert agg in (
            neat_agg.sum_aggregation,
            neat_agg.max_aggregation,
            neat_agg.mean_aggregation
        ), "Unknown aggregation"

    return _cppneat.CPPNet(
        list(input_nodes),
        list(output_nodes),
        list(node_evals)
    )

class BatchEvaluator:
    def __init__(self, data_cache: Dict[str, DataSlice], num_look_back_period: int = 10):
        self._data_cache = data_cache
        self._n = num_look_back_period
        self._files = list(data_cache.keys())
        
        self._max_episode_length = 390 - self._n
        
        # Reward calculation parameters
        self._illegal_action_penalty = -5 / 1000
        self._per_side_fee_rate = 0.0003
        self._end_action_time_delta = 389.0
        
        # Observation scaling factors
        self._time_scale_factor = 1e-2
        self._stock_change_scale_factor = 1e3
        self._volume_scale_factor = 1
        self._close_to_mid_ratio_scale_factor = 1
        
        # Sharpe ratio parameters
        per_year_bond_yield = 5
        trading_days = 250.0
        self._risk_free_rate = (1 + per_year_bond_yield/100)**(1/trading_days) - 1
        self._epsilon = 1e-6
        
        # One-hot encoding for actions
        self._one_hot_map = np.array([
            [1, 0, 0],      # neutral
            [0, 1, 0],      # short
            [0, 0, 1],      # long
        ])

    def reset(self):
        """Initialize environment state for a new episode."""
        # same file is used for the entire population
        self._FILE_NAME = np.random.choice(self._files)
        self._file_df = self._data_cache[self._FILE_NAME]
        self._start_ind = 0
        
        # Initialize position state
        self._load_data_slice() # initializes stock_changes, volume_changes, market_start_time_delta = n-1, close to mid ratio
        self._prev_action = np.full(shape=(self._batch_size, ), fill_value = SimpleState.neutral.value)
        self._prev_price = np.full(shape = (self._batch_size), fill_value = np.nan)
        self._prev_action_time_delta = np.full(shape = (self._batch_size,), fill_value= self._market_start_time_delta )
        
        # prepare for step function
        self._start_ind += 1
        self._is_episode_over = False
        
        return self._make_state()

    def _load_data_slice(self):
        """Load the current data window and update metrics."""
        start, end = self._start_ind, self._start_ind + self._n
        self._market_start_time_delta = self._file_df.time[end-1]
        
        closes = self._file_df.close[start:end]
        # Calculate price changes
        self._stock_changes = np.log(closes) - np.log(self._file_df.open[start:end])
        
        # Calculate volume changes
        self._volume_changes = np.log(self._file_df.volume[start:end]) - np.log(self._file_df.historical_avg_vol[start:end])
        
        # Calculate close-to-mid ratios
        self._close_to_mid_ratio = np.zeros(self._n, dtype=np.float32)
        highs = self._file_df.high[start:end]
        lows = self._file_df.low[start:end]
        mids = (highs + lows) / 2
        ranges = highs - lows
        np.divide(2 * (closes - mids), ranges, out=self._close_to_mid_ratio, where= (ranges != 0))

    def _make_state(self):
        """
        Build a batch of states of shape (B, 3n+6):
        - 3 dims: one-hot prev_action  (varies per batch element)
        - 1 dim : market time          (same scalar broadcast)
        - 1 dim : prev-action time     (varies per batch element)
        - 1 dim : trend strength       (same scalar broadcast)
        - n dims: scaled stock_changes (same vector broadcast)
        - n dims: scaled volume_changes
        - n dims: scaled close_to_mid
        """
        B = self._batch_size

        # 1) One‑hot prev_action: shape (B,3)
        one_hot = self._one_hot_map[self._prev_action]  # (B,3)

        # 2) Time features:
        #  - market time: scalar → broadcast to (B,1)
        #  - prev_action_time_delta: (B,) → reshape (B,1)
        market_times = np.full((B,1), self._market_start_time_delta, dtype=np.float32)
        last_action_times = self._prev_action_time_delta.reshape(B,1)
        time_feats = np.concatenate([market_times, last_action_times], axis=1)
        time_feats *= self._time_scale_factor    # (B,2)

        # 3) Trend strength: scalar → broadcast to (B,1)
        trend_val = np.sum(self._stock_changes) / np.sum(np.abs(self._stock_changes))
        trend_vec = np.full((B,1), trend_val, dtype=np.float32)

        # 4) Window features (each length‑n) → broadcast to (B,n)
        stock_mat = np.tile(self._stock_changes * self._stock_change_scale_factor, (B,1))
        vol_mat   = np.tile(self._volume_changes * self._volume_scale_factor,     (B,1))
        mid_mat   = np.tile(self._close_to_mid_ratio * self._close_to_mid_ratio_scale_factor, (B,1))

        # 5) Concatenate into final state of shape (B, 3 + 2 + 1 + 3n)
        state_batch = np.concatenate([
            one_hot,   # (B,3)
            time_feats,# (B,2)
            trend_vec, # (B,1)
            stock_mat, # (B,n)
            vol_mat,   # (B,n)
            mid_mat    # (B,n)
        ], axis=1)

        return state_batch

    def step(self, actions: np.ndarray):
        if self._is_episode_over:
            return np.zeros(self._batch_size), True
            
        # Load new data window
        self._load_data_slice()
        self._prev_action_time_delta += 1.0 
        
        # Get current price
        current_price = self._file_df.close[self._start_ind + self._n - 1]
        
        """ Assign Masks for both eod, normal handling, reward calculation """
        # current masks
        short_mask  = (actions == SimpleState.short.value)
        long_mask = (actions == SimpleState.long.value)
        neutral_mask = (actions == SimpleState.neutral.value)
        # previous masks
        prev_neutral_mask = (self._prev_action == SimpleState.neutral.value)
        prev_non_neutral_mask = ~prev_neutral_mask
        prev_long_mask = (self._prev_action == SimpleState.long.value)
        prev_short_mask = (self._prev_action == SimpleState.short.value)
        
        # Sanity: non‑neutral ≡ (long OR short)
        assert np.all(prev_non_neutral_mask == (prev_long_mask | prev_short_mask)), \
       "prev_non_neutral_mask does not match prev_long_mask | prev_short_mask"
        
        """ End-of-day handling """
        if self._market_start_time_delta >= self._end_action_time_delta:
            reward = np.zeros((self._batch_size,), np.float32)
            
            
            # previous long and the gene correctly predicts a neutral
            was_long = prev_long_mask & neutral_mask
            reward[was_long] = np.log( current_price * (1 - self._per_side_fee_rate) / (self._prev_price[was_long] * (1 + self._per_side_fee_rate)) )
            
            # previous short and the gene correctly predicts a neutral
            was_short = prev_short_mask & neutral_mask 
            reward[was_short] = np.log( self._prev_price[was_short] * (1 - self._per_side_fee_rate) / (current_price * (1 + self._per_side_fee_rate)) )
            
            # previous non-neutral (short/long) but the agent predicts non-neutral (hold/short/long) (illegal)
            reward[ prev_non_neutral_mask & ~neutral_mask] = self._illegal_action_penalty
                
            # updates the states
            self._prev_action[prev_non_neutral_mask] = SimpleState.neutral.value    # a neutral action for a previous non-neutral action
            self._prev_action_time_delta[prev_non_neutral_mask] = 0.0               # update the time delta
            self._prev_price[prev_non_neutral_mask] = np.NaN                        # update the previous non-neutral prices
            
            
            # previous neutral but gene tries to open a position (illegal)
            reward[prev_neutral_mask & (long_mask | short_mask)] = self._illegal_action_penalty               
            
            # prvious neutral but the agent tales a neutral action (illegal)
            reward[prev_neutral_mask & neutral_mask] = self._illegal_action_penalty
            
            # previous neutral and the agnet does not open a position, so no penalty
            # reward is preassigned to zero for these cases

            self._is_episode_over = True
            return self._make_state(), reward, True
        
        """ Action Cycling: Assume that the current action is legal but not new. """
        is_legal_action = np.full((self._batch_size,), True, dtype=bool)
        is_new_action = np.full((self._batch_size,), False, dtype=bool)

        # handle short action
        is_new_action[prev_neutral_mask & short_mask] = True  # neutral to short transition
        self._prev_action_time_delta[prev_neutral_mask & short_mask] = 0.0
        
        is_legal_action[prev_non_neutral_mask & short_mask] = False # non-neutral to short illegal transition
        
        # handle long action
        is_new_action[prev_neutral_mask & long_mask] = True # neutral to long transition (legal)
        self._prev_action_time_delta[prev_neutral_mask & long_mask] = 0.0
        
        is_legal_action[prev_non_neutral_mask & long_mask] = False # non-neutral to long transition (illegal)
            
        # handle neutral action
        is_new_action[prev_non_neutral_mask & neutral_mask] = True  # non-neutral to neutral transition (legal)
        self._prev_action_time_delta[prev_non_neutral_mask & neutral_mask] = 0.0
        
        is_legal_action[prev_neutral_mask & neutral_mask] = False # neutral to neutral transition (illegal)
                
        # Hold action is always legal, but never new
        
        """ Reward Calculation: All new actions are legal by default, but not all legal actions are new, ie holds.
                                Hold is a legal but not new action."""
        
         # sanity: no new action should ever be illegal
        assert np.all(is_legal_action[is_new_action]), \
            "Found new actions that are marked illegal!"
        
        reward = np.zeros((self._batch_size,), np.float32)
        
        # Handle illegal action
        actions[~is_legal_action] = SimpleState.hold.value       
        reward[~is_legal_action] = self._illegal_action_penalty
        
        # Handle new actions. By default all new actions are legal.
        # set the price when initiating trade, ie long and short
        self._prev_price[is_new_action & (short_mask | long_mask)] = current_price 
        
        # when closing a trade assign the reward
        close_mask = is_new_action & neutral_mask
        if np.any(close_mask):
            was_long  = close_mask & prev_long_mask
            was_short = close_mask & prev_short_mask

            # longs: entered at prev_price, exit at scalar current_price
            reward[was_long] = np.log( current_price * (1 - self._per_side_fee_rate) / (self._prev_price[was_long] * (1 + self._per_side_fee_rate)) )

            # shorts: entered at prev_price, cover at current_price
            reward[was_short] = np.log( self._prev_price[was_short] * (1 - self._per_side_fee_rate) / (current_price * (1 + self._per_side_fee_rate)) )

            # reset prev_price for closed positions
            self._prev_price[close_mask] = np.nan
            
            
        # Update state for new actions (except closing which is handled above)
        self._prev_action[is_new_action] = actions[is_new_action] # holds are not new actions
        
        state = self._make_state()
        # Prepare for next step
        self._start_ind += 1
        
        return state, reward, False

    def call_internal(self, genomes, config):
        """Batch‐evaluate genomes and assign fitness in place."""
        B = len(genomes)
        self._batch_size = B

        # 1) Build networks
        nets = [neat.nn.FeedForwardNetwork.create(g, config) for _, g in genomes]
        cpp_nets = [build_cppnet(net.input_nodes, net.output_nodes, net.node_evals,) for net in nets]
            
            
        # 2) Pre‑allocate reward history
        returns = np.full((B, self._max_episode_length), np.nan, dtype=float)
        step_count = 0

        # 3) Roll out one episode
        state =  self.reset()
        done = False
        while not done:
            outputs = [ cpp_nets[i].activate(state[i]) for i in range(B)]
            
            outputs_arr = np.array(outputs)  # shape (B, output_dim)
            actions = np.argmax(outputs_arr, axis=1)
            
            state, rewards, done = self.step(actions)
            returns[:, step_count] = rewards
            step_count += 1

        # 4) Vectorized Sharpe computation
        means = np.nanmean(returns, axis=1)                   # shape (B,)
        stds  = np.nanstd(returns, axis=1, ddof=0)            # shape (B,)
        sharpes = (means - self._risk_free_rate) / (stds + self._epsilon)

        # 5) Assign fitness back to genomes
        for (_, genome), s in zip(genomes, sharpes):
            # If no valid steps or zero‐volatility, give -inf
            if np.isnan(s):
                genome.fitness = -np.inf
            else:
                genome.fitness = s
                
    def __call__(self, genomes, config):
        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()
        self.call_internal(genomes, config)
        pr.disable()

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
        ps.print_stats(30)
        print(s.getvalue())
