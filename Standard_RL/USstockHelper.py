from os import path,makedirs
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from plotly import graph_objects as go
import bisect
import regex as re
from typing import List, NamedTuple
from enum import Enum
import matplotlib.pyplot as plt
from IPython.display import clear_output
from pathlib import Path
import numpy as np


DATA_DIR = "../US_Market_Data"
interval="1m"
utc_to_ny = timedelta(hours=-4)
date_format = "%Y-%m-%d"
date_file = "date_file.txt"

FILE_NAME:str 

class PrintTypes(Enum):
    normal = 1
    buy = 2
    sell = 3

class DataSlice(NamedTuple):
    time: float = None
    open: float = None
    high: float = None
    low: float = None
    close : float = None
    volume : int = None

# store plot kwargs instead of a format string
Data_dict = {
    PrintTypes.normal: {
        "x_val": [],
        "y_val": [],
        "plot_kwargs": {"c": "b", "marker": ".", "s":  20}  # small blue dot
    },
    PrintTypes.buy: {
        "x_val": [],
        "y_val": [],
        "plot_kwargs": {"c": "g", "marker": "o", "s": 200}  # big green circle
    },
    PrintTypes.sell: {
        "x_val": [],
        "y_val": [],
        "plot_kwargs": {"c": "r", "marker": "o", "s": 200}  # big red circle
    }
}

Money_Made: int


def optional_dir_create(ticker):
    global DATA_DIR
    dir_path = f"{DATA_DIR}/{ticker}"
    
    if not path.exists(dir_path):
        makedirs(dir_path)
        file_path = f"{dir_path}/{date_file}"
        file = open(file_path,"x")
        file.close()

def write_date(date_str: str, ticker: str):
    global DATA_DIR, date_format
    file_path = f"{DATA_DIR}/{ticker}/{date_file}"

    # Parse the new date
    curr_date = datetime.strptime(date_str, date_format)

    # Read & clean up existing (sorted) dates
    with open(file_path, "r") as f:
        cleaned = (line.strip() for line in f)
        date_list = [
            datetime.strptime(l, date_format)
            for l in cleaned
            if l  # skip any blank lines
        ]

    # Insert if not already present
    if curr_date not in date_list:
        bisect.insort(date_list, curr_date)

        # Write updated list back
        with open(file_path, "w") as f:
            for dt in date_list:
                f.write(dt.strftime(date_format) + "\n")
                     
def dataframe_csv_write(df, ticker, date_str):
    global DATA_DIR
    file_path = f"{DATA_DIR}/{ticker}/{date_str}"
    pd.DataFrame.to_csv(df, file_path, mode="w", header=True, index_label="Time")
    write_date(date_str,ticker)
        
def create_previous_data_file(tickers, date_str):
    """
    Downloads intraday price data for given tickers on a specific date,
    processes the data by rounding prices and formatting the time,
    and writes the result to CSV files, one per ticker.

    Parameters:
    - tickers (list of str): List of stock ticker symbols.
    - date_str (str): Date string in 'YYYY-MM-DD' format for which to download data.
    """
    global interval, utc_to_ny, date_format
    
    # Parse input date string
    date_obj = datetime.strptime(date_str, date_format)

    # Set start and end range for the date (1 day interval)
    start = date_obj.strftime(date_format)
    end = (date_obj + timedelta(days=1)).strftime(date_format)

    # Download intraday stock data using yfinance
    df = yf.download(tickers, start=start, end=end, interval=interval)
    price_cols = ['Close', 'High', 'Low', 'Open']

    for ticker in tickers:
        # Extract the specific ticker's DataFrame
        local_df = df.xs(ticker, axis=1, level=1)

        # Create directory if not exists
        optional_dir_create(ticker)
        
        # handle missing data by a simple linear interpolation
        local_df = local_df.interpolate(method="linear")
        # side effect of interpolation, round to the nearest integer
        local_df.loc[:,"Volume"] = local_df.loc[:,"Volume"].round()

        # Format index to only keep hour and minute (e.g., '13:45') in ny time zone
        local_df.index += utc_to_ny
        local_df.index = local_df.index.strftime('%H:%M')

        # Round all price columns to 2 decimal places
        local_df.loc[:, price_cols] = local_df[price_cols].round(2)
        
        local_df = update_dataframe(local_df)

        # Save the processed data to a CSV
        dataframe_csv_write(local_df, ticker, date_str)

def printStaticGraph(date_str:str, symbol:str):
    """
     Given the symbol and the dateFile of that symbol, this function prints the graph corresponding to the stock price on that particular day.

    Args:
        date_str (str): YYYY-MM-DD
        symbol (str): eg AAPL
    """
    filepath = f"{DATA_DIR}/{symbol}/{date_str}"
    
    df = pd.read_csv(filepath, index_col=0, header=0)
    
    fig = go.Figure(data=[go.Candlestick( x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Candlesticks" )])
    
    # Volume chart (bar chart)
    fig.add_trace(go.Bar( x=df.index, y=df['Volume'], name="Volume", marker=dict(color='rgba(0, 0, 255, 0.3)'), yaxis='y2'))
    
    # Update layout
    fig.update_layout( title= f"{symbol} on {date_str}", xaxis_title="Time", yaxis_title="Price", yaxis2=dict(title="Volume",overlaying="y",side="right"), xaxis_rangeslider_visible = False)

    fig.show(renderer="notebook")   

def printDynamicGraph(x_val_new: List, y_val_new: List, 
                      print_type: PrintTypes = PrintTypes.normal, money_made = None):
    
    global FILE_NAME, Money_Made
    # update the global Money_Made
    if money_made is not None:Money_Made = money_made
    # accumulate
    Data_dict[print_type]["x_val"].extend(x_val_new)
    Data_dict[print_type]["y_val"].extend(y_val_new)

    # clear the old output & figure
    clear_output(wait=True)
    # re-plot everything
    plt.figure(figsize=(10, 6))
    for pt, info in Data_dict.items():
        x = info["x_val"]
        y = info["y_val"]
        kw = info["plot_kwargs"]
        if x and y:
            plt.scatter(x, y, label=pt.name, **kw)

    plt.legend()
    plt.title(f"Agent Analysis for: {FILE_NAME}, Money_Made:{Money_Made}")
    plt.xlabel("Time Since Market Start (Mins)")
    plt.ylabel(f"Stock Price ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
 
def resetDynamicGraph(file_name:str):
    global Data_dict, FILE_NAME, Money_Made
    Money_Made = 0.0
    FILE_NAME = file_name
    plt.clf()
    Data_dict = {
        PrintTypes.normal: {
            "x_val": [],
            "y_val": [],
            "plot_kwargs": {"c": "b", "marker": ".", "s":  20}  # small blue dot
        },
        PrintTypes.buy: {
            "x_val": [],
            "y_val": [],
            "plot_kwargs": {"c": "g", "marker": "o", "s": 200}  # big green circle
        },
        PrintTypes.sell: {
            "x_val": [],
            "y_val": [],
            "plot_kwargs": {"c": "r", "marker": "o", "s": 200}  # big red circle
        }
    }

def check_datefile_accuracy(dir_str: str):
    """
    Validates the consistency between files listed in 'date_file.txt' and actual files present 
    in the specified directory.

    The function performs two checks:
    1. Ensures that every file listed in 'date_file.txt' exists in the directory.
    2. Ensures that every file present in the directory (excluding 'date_file.txt' itself) 
       is listed in 'date_file.txt'.

    Any mismatches are printed to the console.

    Args:
        dir_str (str): 
            Path to the target directory, relative to the 'Analysis' folder. 
            Assumes 'date_file.txt' exists in this directory.

    Prints:
        Messages indicating files that are missing from either the directory or 'date_file.txt'.
    """
    dir_path = Path(dir_str)
    
    date_file_data = []
    directory_data = []
    
    with open(f"{dir_str}/date_file.txt",'r') as file:
        for line in file:
            date_file_data.append(line.strip()) # remove the newline character
    
    for file in dir_path.rglob("*"):
        if file.name == "date_file.txt" or file.is_dir():continue
        directory_data.append(file.name)
    
    # check datefile data in directory
    for file in date_file_data:
        if file in directory_data: continue
        print(f"{file} found in date_file but not in given directory {dir_str}")
    
    # check directory data in datefile
    for file in directory_data:
        if file in date_file_data: continue
        print(f"{file} found in {dir_str} but not in date_file")
        
def update_dataframe(df):
    # define trading window
    trading_start = datetime.strptime('09:30','%H:%M')
    trading_end = trading_start + timedelta(minutes=389)
    # build df_out with time strings index
    times = [(trading_start + timedelta(minutes=i)).strftime('%H:%M') for i in range(390)]
    df_out = pd.DataFrame(index=times, columns=df.columns, dtype=float)
    df_out.index.name = "Time"
    # parse df.index strings or datetimes into sorted list of time objects
    # assume df.index is sorted and in "HH:MM"
    times_data = [datetime.strptime(t, '%H:%M') for t in df.index]
    # fill start
    first = times_data[0]
    if first > trading_start:
        delta = int((first - trading_start).total_seconds()//60)
        for i in range(delta+1):
            t = (first - timedelta(minutes=i)).strftime('%H:%M')
            df_out.loc[t] = df.iloc[0]
    else:
        df_out.loc[df.index[0]] = df.iloc[0]
    # fill between pairs
    for i in range(len(df.index)-1):
        t0 = times_data[i]; t1 = times_data[i+1]
        idx0 = df.index[i]; idx1 = df.index[i+1]
        df_out.loc[idx0] = df.loc[idx0]
        gap = int((t1 - t0).total_seconds()//60) - 1
        if gap == 0:continue
        prev = df.loc[idx0]; nex = df.loc[idx1]
        if gap == 1:
            t = (t0 + timedelta(minutes=1)).strftime('%H:%M')
            open_i = prev["Close"]
            close_i = nex["Open"]
            df_out.loc[t, ["Open","High","Low","Close","Volume"]] = [
                open_i,
                max(open_i, close_i),
                min(open_i, close_i),
                close_i,
                (prev["Volume"] + nex["Volume"])/2
            ]
        else:
            closes = np.linspace(prev["Close"], nex["Open"], gap+2)
            vols = np.linspace(prev["Volume"], nex["Volume"], gap+2)
            for k in range(1, gap+1):
                t = (t0 + timedelta(minutes=k)).strftime('%H:%M')
                open_i = closes[k-1]
                close_i = closes[k]
                df_out.loc[t, ["Open","High","Low","Close","Volume"]] = [
                    open_i,
                    max(open_i, close_i),
                    min(open_i, close_i),
                    close_i,
                    vols[k]
                ]
    # fill last
    last = times_data[-1]
    if last < trading_end:
        delta = int((trading_end - last).total_seconds()//60)
        for i in range(delta+1):
            t = (last + timedelta(minutes=i)).strftime('%H:%M')
            df_out.loc[t] = df.iloc[-1]
    else:
        df_out.loc[df.index[-1]] = df.iloc[-1]
    # ensure last row copied
    df_out = df_out.sort_index()
    df_out.loc[df_out["Volume"] == 0, "Volume"] = 1

    return df_out