from enum import Enum
from tf_agents.environments import PyEnvironment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np
from USstockHelper import DataSlice
from typing import  Optional,  Dict, List
import pandas as pd
from pathlib import Path
import random
from typing import NamedTuple
import scipy.stats as scipy_stats


    
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
            
            # Convert all ohlc data to representative space, and volume to relative_volume
            market_open = curr_slice.open[0]
            
            new_cache[file]= DataSlice(
                time=curr_slice.time,
                open= curr_slice.open/market_open,
                close=curr_slice.close/market_open,
                low = curr_slice.low/market_open,
                high=curr_slice.high/market_open,
                volume=curr_slice.volume/rel_volume
            )
            
            done_files.add(file)
    return cache if num_prev_file ==0 else new_cache

class StockState(float, Enum):
    short = -1.0
    noTrade = 0.0
    long = 1.0


class USStockEnv(PyEnvironment):
    
    def __init__(self, data_cache: Dict[str, DataSlice], reward_scale_factor:float=1, look_back_period = 10):
        """
            Initialize the trading environment.

            Args:
                data_cache (Dict[str, DataSlice]):
                    Mapping from file path (relative to dataset root) to a DataSlice object containing
                    OHLCV data and time deltas (in minutes) for each bar/candle.

                reward_scale_factor (float, default=1):
                    Scalar applied to the natural log-return reward. Chosen such that the mean absolute
                    reward magnitude across the dataset is approximately 1 for stable training.

                look_back_period (int, default=10):
                    Number of past time steps to include in the sliding observation window when 
                    constructing each environment state.
            """
        super().__init__(handle_auto_reset= False) # This environment will handle resets not base class
        
        
        # agent action spec from 0->2 and internal map for conversion
        self._action_spec = array_spec.BoundedArraySpec( shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self._action_map = { 0: StockState.short, 1: StockState.noTrade, 2: StockState.long}
        
        # agent simple features (independent and dependent) and compound features
        self._L = look_back_period
        self._I = 6        # agent independent features
        self._D = 4         # agent dependent features
        self._C = 6         # compound features
        self._observation_spec = {
            'independent': array_spec.ArraySpec( shape=(self._L, self._I), dtype=np.float32, name='ind_obs'),
            'dependent': array_spec.ArraySpec(shape=(self._L, self._D), dtype=np.float32, name='dep_obs'),
            'compound': array_spec.ArraySpec( shape=(self._C,), dtype=np.float32, name='comp_obs'),
        }
        
        # Some Variables
        self._market_end_time_delta = 389.0
        self._data_cache = data_cache
        self._files = list(self._data_cache.keys())
        self._FILE_NAME = None # current file for logging purposes
        self._Holding_Threshold = self._L 
        
        """ Scale Factors """
        # reward scale factor
        self._reward_sf               = reward_scale_factor
        
        # Compound states scale factor
        self._skew_sf       =   1/1.5     # 1/resolution
        self._kurtotsis_sf  =   1/5
        self._max_dd_sf     =   self._reward_sf / 4 # eye ball estimate
        self._volatility_sf =   self._reward_sf
        
        # Dependent states scale factor
        self._cumul_pnl_sf       =   self._reward_sf / 10
        self._mr_action_time_delta_sf   =   1/self._market_end_time_delta      # a reasonable upper bound on holding time

        # Independent states scale factors
        self._true_range_sf             =   500
        self._intra_bar_vol_porxy_sf    =   self._reward_sf
        self._dlogprice_sf              =   self._reward_sf
        self._ohlc_sf                   =   self._reward_sf/10
        # dlogvolume, close_to_mid_ratios, time_completion_ratios need not be scaled
    
        # hindight bonus weighting factor
        self._hindsight_wf =        1
        
        """ Fees """
        # a single side fee rate that is applied individually on both buy and sell
        self._one_side_fee      =   0.0003
        self._log_one_side_fee  =   np.log(1 - self._one_side_fee)
        
        """ Predefined Rewards """
        # To promote trading and demote inactivity
        self._inactive_wf       =   0.0 # inactive weighting period
        self._inactive_k        =   2   # k = 1, exponential from 0 to 1, k>1:initially slow, then fast, k<1:initally fast, then slow
        self._max_inactive_time =   25  # [minutes] above this, the max penalty of lambda * new_reward will be charged.      
        
        """ logging purposes """
        self.logger: Optional[List[Dict]] = None
        self.logging_step: Optional[int] = None
        
        self._reset()

    def action_spec(self): return self._action_spec
    
    def observation_spec(self): return self._observation_spec
    
    def reward_spec(self):
        return {
            "training":  array_spec.ArraySpec(shape=(), dtype=np.float32, name='reward'),
            "evaluation":  array_spec.ArraySpec(shape=(), dtype=np.float32, name='evaluation'),
        }
    
    def _reset(self):
        # Internal States
        self._FILE_NAME = random.choice(self._files)
        self._file_df = self._data_cache[self._FILE_NAME]
        self._start_ind = 0
        ranges = self._get_short_slice(self._file_df, self._start_ind, self._L)
        self._market_open = ranges.open[0]
        self._holding_period = 0
        
        self._pos = StockState.noTrade  # Define the initial state, assuming it adheres to the observation spec
        self._unreal_cumul = 0.0        # unrealized cumulative training reward
        self._real_cumul = 0.0          # realized cumulative training reward
        self._train_reward = 0.0
        
        
        """ Create the reset state """
        # Set previous stock value and time based on last available stocks and times
        self._market_start_time_delta = ranges.time[self._L - 1]   # L-1 minutes since market start
        self._mr_action_time_delta = ranges.time[self._L - 1]   #  L-1 minutes since market start
        
        independent_state = self._make_independent_state(ranges)
        compound_state = self._make_compound_state(ranges)
        
        """ 
        Dependent State:
        0. MR time delta is an array of times since the market start, even though no actions have taken place.
        1. MR direction is 0, ie no Trade which is explicity assigned. 
        2. unrealized cumul pnl for curr trade is 0 since no trades have happened yet
        3. realized cumul pnl for all trades is 0 since not trades have happened yet
        """

        self._dependent_state = np.zeros(shape=(self._L, self._D), dtype=np.float32)
        self._dependent_state[:,0] = ranges.time / self._market_end_time_delta
        self._dependent_state[:,1] = self._pos.value
        
        # Prepration for step function
        self._is_episode_over = False
        self._start_ind +=1

        # some safety checks
        assert independent_state.shape == (self._L, self._I)
        assert compound_state.shape == (self._C,)

        # Return the initial time step with the state
        return ts.restart( observation={'independent': independent_state, 'dependent': self._dependent_state, 'compound':compound_state},
                          reward_spec=self.reward_spec())

    def _step(self, action):
        if self._is_episode_over : return self._reset()
        pos_new = self._action_map[int(action)]
        ranges = self._get_short_slice(self._file_df, self._start_ind, self._L)
        
        """ Update metrics based on new data """
        # calculate new start time delta and prev time delta
        # the difference between previous start prediction time delta and current latest time is an interval, so add that 
        self._mr_action_time_delta += ranges.time[-1] - self._market_start_time_delta
        self._market_start_time_delta = ranges.time[-1]

        new_stock_price = ranges.close[-1] # new representative stock price
        new_reward = np.log(new_stock_price) - np.log(ranges.open[-1]) # stock change,
        
        # create independent simple and compound states
        independent_state = self._make_independent_state(ranges)
        compound_state = self._make_compound_state(ranges)
        
        """ Predefine some states """
        train_raw = 0.0
        eval_reward = 0.0
        
        """ EOD handling """
        if self._market_start_time_delta >= self._market_end_time_delta:
            # EOD does not take agent action into account for penalization
            
            if self._pos != StockState.noTrade:  # currently holding, so force a closure
                eval_reward = new_stock_price * (self._pos.value - self._one_side_fee) # evaluation reward
                
                self._real_cumul += self._unreal_cumul + self._log_one_side_fee
                state_unreal_cumul = self._log_one_side_fee
                self._train_reward = self._log_one_side_fee # normal case reward, just the fee
                self._mr_action_time_delta = 0.0
                
            else: #then no fees, use previous time delta, 
                state_unreal_cumul = 0.0
            
            # Force the agent to close the trade
            pos_new = StockState.noTrade
            
            self._dependent_state = self._make_dependent_state(self._dependent_state, self._mr_action_time_delta, pos_new, state_unreal_cumul, self._real_cumul)
            
            # log states before reward scaling
            if self.logger is not None and self.logging_step is not None:
                self.logger.append({
                    'step':  self.logging_step,
                    'raw':   train_raw,
                    'reward': self._train_reward,
                    'eval_reward':eval_reward,
                    'action': pos_new.name,
                    'prev_action': self._pos.name,
                    'data': new_stock_price,
                })
            # prepare for reset
            self._is_episode_over = True 
            return ts.termination({'independent': independent_state, 'dependent': self._dependent_state, 'compound':compound_state},
                                  reward = {"training": self._train_reward * self._reward_sf, "evaluation": eval_reward * self._reward_sf})

        """ Reward Calculation """
        # precompute change in position
        delta_pos = self._pos.value - pos_new.value
        
        # literal money reward
        eval_reward = new_stock_price * ( delta_pos - abs(delta_pos) * self._one_side_fee ) # stock price - fees
        
        # cumulative-EMA reward
        train_raw = pos_new.value * new_reward
        train_fees = abs(delta_pos) * self._log_one_side_fee
        
        if pos_new != self._pos:
            self._real_cumul += self._unreal_cumul + train_fees
            self._unreal_cumul = 0.0
            self._holding_period = 0
            self._mr_action_time_delta = 0.0
        
        self._unreal_cumul += train_raw
        
        if pos_new != StockState.noTrade: self._holding_period += 1
        
        if self._holding_period > self._Holding_Threshold:
            alpha = 0.2
            self._train_reward = (1 - alpha) * self._train_reward + alpha * train_raw
        else:
            self._train_reward = self._unreal_cumul / self._Holding_Threshold
        
        state_unreal_cumul = self._unreal_cumul + train_fees # for dependent state calculation
        self._train_reward += train_fees
        
        # inactive penalty computation
        inactive_penalty = 0.0
        if self._pos == pos_new and pos_new == StockState.noTrade: # holding on noTrade
            x = min(self._mr_action_time_delta/self._max_inactive_time, 1) # cap x at 1
            inactive_penalty = -((2 ** x - 1) ** self._inactive_k) * np.abs(new_reward)
                
        # combining all rewards        
        self._train_reward +=  self._inactive_wf * inactive_penalty
    
        # get the new dependent state
        self._dependent_state = self._make_dependent_state(self._dependent_state, self._mr_action_time_delta, pos_new, state_unreal_cumul, self._real_cumul)
        
        # log states before reward scaling
        if self.logger is not None and self.logging_step is not None:
            self.logger.append({
                'step':  self.logging_step,
                'raw':   train_raw,
                'reward': self._train_reward,
                'eval_reward':eval_reward,
                'action': pos_new.name,
                'prev_action': self._pos.name,
                'data': new_stock_price,
            })
        
        # update current position for next iteration
        self._pos = pos_new
        self._start_ind += 1
        return ts.transition(observation = {'independent': independent_state, 'dependent': self._dependent_state, 'compound':compound_state},
                            reward = {"training": self._train_reward * self._reward_sf, "evaluation": eval_reward * self._reward_sf})

    def _make_independent_state(self,data_slice: DataSlice):
        """
        Construct a single state for L timestep. Assumes that the data_slice has L window ohlcv data

        independent (agent-independent states) :
            Close normalized: log(close/market_open)
            Intra Bar Volatility Proxy: log(high/low)
            Dlogprice: log(close/open) for the current time step
            Time Completion Ratio: Time since market start / total market time
            DlogVolume: log(volume/historical_median_volume) for the current time step
            Close to Mid ratio: 2*(close - (high + low)/2)/(high - low) for the current time step,  
                If high = low, then this number is 0
        """
        # Close Norm
        close_normalized = (np.log(data_slice.close) - np.log(self._market_open)) * self._ohlc_sf  
        
        # Intra Bar vol
        intra_bar_vol_proxy = (np.log(data_slice.high) - np.log(data_slice.low) ) * self._intra_bar_vol_porxy_sf
        
        # DLogPrice
        dlogprice = (np.log(data_slice.close) - np.log(data_slice.open) ) * self._dlogprice_sf
        
        # Time Completion Ratios
        time_completion_ratios = data_slice.time/ self._market_end_time_delta
        
        # DlogVolume
        dlogvolume = np.log(data_slice.volume) # volume is already volume / historical average volume
        
        # Close to Mid Ratio
        num = 2*(data_slice.close - (data_slice.high + data_slice.low)/2)
        den = data_slice.high - data_slice.low
        ratio = np.zeros_like(num, dtype=np.float32)
        close_to_mid_ratios = np.divide(num, den, out=ratio, where=(den != 0))
        
        return np.stack([
            close_normalized,
            intra_bar_vol_proxy,
            dlogprice,
            time_completion_ratios,
            dlogvolume,
            close_to_mid_ratios,
            ],
            axis = -1, # stack along the feature axis, not the lookback axis
            dtype= np.float32
        )
    
    def _make_dependent_state(self, prev_dependent_state: np.ndarray, mr_time_delta: float, mr_dir: StockState, unreal_cumul:float, real_cumul:float):
        """
        Slide and update the agent-dependent state window.

        The dependent-state window has shape (L, D) with columns:
            0: most-recent-action-time-ratio (time since most recent action / total market time)
            1: most-recent-direction (−1, 0, +1)
            2: Unrealized Cumulative Reward (float; computed in step function)
            3: Realized Cumulative Reward (float)
            
        Notes:
        - This implementation updates `prev_dependent_state` in-place (returns the same array).
        - Expects `prev_dependent_state` to have at least 3 columns and L rows.
        - `mr_time_delta` is divided by the environment session length (self._market_end_time_delta).
        """
        
        assert prev_dependent_state.shape == (self._L, self._D)
        new_states = prev_dependent_state

        new_states[0:self._L - 1, :] = new_states[1:self._L, :]

        new_states[-1, 0] = np.float32(mr_time_delta * self._mr_action_time_delta_sf)
        new_states[-1, 1] = np.float32(mr_dir.value)
        new_states[-1, 2] = np.float32(unreal_cumul * self._cumul_pnl_sf)
        new_states[-1, 3] = np.float32(real_cumul * self._cumul_pnl_sf)
        

        return new_states

    def _make_compound_state(self, data_slice: DataSlice):
        """
        Compute compound (window-level) features from an L-length DataSlice.

        Returns a 1-D float32 array of length 6 with the following entries:
        price_trend                 = sum(dlogprice) / sum(|dlogprice|)
        unscaled_price_momemtum     = sum(|dlogprice|)
        volume_trend                = sum(dlogvolume) / sum(|dlogvolume|)
        skew                        = sample skewness of dlogprice (bias=False)
        kurtosis                    = sample Fischer kurtosis of dlogprice (normal=3, bias=False)
        max_log_dd                  = maximum log drawdown over the window: max_{i<=j} log(high_i/low_j)

        All logs are computed with a small epsilon to avoid -inf/nan. Returns dtype np.float32.
        """
        dlogprice = np.log(data_slice.close) - np.log(data_slice.open)
        dlogvolume = np.log(data_slice.volume)
        
        # Price Trend
        price_trend = np.sum(dlogprice)/np.sum(np.absolute(dlogprice))
        
        # Unscaled momemtum
        unscaled_price_momemtum = np.sum(np.absolute(dlogprice))
        
        # Volume Trend
        volume_trend = np.sum(dlogvolume)/np.sum(np.absolute(dlogvolume))
        
        # skew (sample skew)
        skew = scipy_stats.skew(dlogprice, bias=False) * self._skew_sf # sample skew
        
        # kurtosis (sample Fischer Kurtosis)
        kurtosis = scipy_stats.kurtosis(dlogprice, fisher=True, bias=False)  * self._kurtotsis_sf # fisher kurtosis

        # max drawdown
        log_high = np.log(data_slice.high) 
        log_low = np.log(data_slice.low) 
        cumulative_max = np.maximum.accumulate(log_high) 
        drawdowns = cumulative_max - log_low 
        max_dd = np.max(drawdowns) * self._max_dd_sf
        
        return np.array([
            price_trend,
            unscaled_price_momemtum,
            volume_trend,
            skew,
            kurtosis,
            max_dd
            ], dtype=np.float32)
    
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
        )

    def update_data_cache(self,new_cache: Dict, new_files:List):
        self._data_cache = new_cache
        self._files = new_files
         
