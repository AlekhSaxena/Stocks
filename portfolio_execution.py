import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import copy
import matplotlib.pyplot as plt



from stocktrends import Renko
import statsmodels.api as sm





def CAGR(DF,f):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["cum_return"] = (1 + df["mon_ret"]).cumprod()
    n = len(df)/f
    CAGR = (df["cum_return"].tolist()[-1])**(1/n) - 1
    return CAGR

def volatility(DF,f):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    vol = df["mon_ret"].std() * np.sqrt(f)
    return vol

def sharpe(DF,rf,f):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df,f) - rf)/volatility(df,f)
    return sr
    

def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    df["cum_return"] = (1 + df["mon_ret"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd


def download_historical_data(tickers, start, end, interval):
    """
    Download historical OHLC data for a list of tickers.

    :param tickers: List of ticker symbols.
    :param start: Start date for the data.
    :param end: End date for the data.
    :param interval: Data interval. Default is '1mo' for monthly data.
    :return: Dictionary with tickers as keys and OHLC data as values.
    """
    ohlc_data = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, start, end, interval=interval)
            if not data.empty:
                ohlc_data[ticker] = data.dropna(how="all")
        except Exception as e:
            print(f"Failed to download data for {ticker}: {e}")
    return ohlc_data


def calculate_monthly_returns(ohlc_data):
    """
    Calculate monthly returns for each stock.

    :param ohlc_data: Dictionary with tickers as keys and OHLC data as values.
    :return: DataFrame with monthly returns for each ticker.
    """
    ohlc_dict = copy.deepcopy(ohlc_data)
    return_df = pd.DataFrame()

    for ticker in ohlc_dict.keys():
        print(f"Calculating monthly return for {ticker}")
        ohlc_dict[ticker]['mon_ret'] = ohlc_dict[ticker]['Adj Close'].pct_change()
        return_df[ticker] = ohlc_dict[ticker]['mon_ret']
    return return_df


NSE=pd.read_excel('/Users/alekhsaxena/Downloads/MCAP31032023_0.xlsx')
NSE['ticker'] = NSE['Symbol'] + '.NS'
tickers = NSE[0:150]['ticker']


# Define your list of tickers, start and end dates
#tickers = ['IRFC.NS', 'PFC.NS', 'ADANIPOWER.NS', 'NMDC.NS', 'PNB.NS']

#tickers=['ADANIGREEN.NS', 'ATGL.NS', 'IRFC.NS', 'SAIL.NS', 'FLUOROCHEM.NS', 'BEL.NS', 'IRCTC.NS', 'ADANIPORTS.NS', 'TATAPOWER.NS', 'PNB.NS']
#tickers=['PFC.NS', 'ADANIPOWER.NS', 'NMDC.NS', 'PNB.NS', 'IOB.NS', 'ADANIGREEN.NS', 'SOLARINDS.NS', 'COALINDIA.NS', 'TRENT.NS', 'UNIONBANK.NS', 'NTPC.NS', 'ATGL.NS', 'IOC.NS', 'GAIL.NS', 'JSWENERGY.NS', 'VBL.NS', 'BAJAJ-AUTO.NS', 'HINDPETRO.NS', 'HAL.NS', 'TATAPOWER.NS']
##tickers=bhav['ticker'].tolist()

start_date = dt.datetime.today() - dt.timedelta(365*2)
end_date = dt.datetime.today()
historical_data = download_historical_data(tickers, start_date, end_date, interval='1d')
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)



def calculate_momentum(ohlc_data_dict):
    """
    Calculate 3-month, 6-month, and 9-month momentum for each stock in the ohlc_data_dict.

    :param ohlc_data_dict: Dictionary with tickers as keys and OHLC data as values.
    :return: DataFrame with 'Ticker', 'Date', '3 month momentum', '6 month momentum', and '9 month momentum'.
    """
    results = []

    for ticker, ohlc_data in ohlc_data_dict.items():
        # Ensure data is sorted by date
        ohlc_data = ohlc_data.sort_index()

        # Calculate momentum for 3, 6, and 9 months
        ohlc_data['3m_momentum'] = ohlc_data['Adj Close'].pct_change(periods=3)
        ohlc_data['6m_momentum'] = ohlc_data['Adj Close'].pct_change(periods=6)
        ohlc_data['9m_momentum'] = ohlc_data['Adj Close'].pct_change(periods=9)

        # Get the latest momentum values and the corresponding date
        latest_date = ohlc_data.index[-1]
        latest_3m_momentum = ohlc_data['3m_momentum'].iloc[-1]
        latest_6m_momentum = ohlc_data['6m_momentum'].iloc[-1]
        latest_9m_momentum = ohlc_data['9m_momentum'].iloc[-1]

        # Append results
        results.append({
            'Ticker': ticker, 
            'Date': latest_date, 
            '3 month momentum': latest_3m_momentum, 
            '6 month momentum': latest_6m_momentum, 
            '9 month momentum': latest_9m_momentum
        })

    return pd.DataFrame(results)
momentum_df = calculate_momentum(historical_data)



def calculate_volatility(ohlc_data_dict):
    """
    Calculate 3-month, 6-month, and 9-month volatility for each stock in the ohlc_data_dict.

    :param ohlc_data_dict: Dictionary with tickers as keys and OHLC data as values.
    :return: DataFrame with 'Ticker', 'Date', '3 month volatility', '6 month volatility', and '9 month volatility'.
    """
    results = []

    for ticker, ohlc_data in ohlc_data_dict.items():
        # Ensure data is sorted by date
        ohlc_data = ohlc_data.sort_index()

        # Calculate daily returns
        ohlc_data['daily_return'] = ohlc_data['Adj Close'].pct_change()

        # Calculate rolling standard deviation (volatility) for 3, 6, and 9 months
        # Assuming each month has roughly 21 trading days
        ohlc_data['3m_volatility'] = ohlc_data['daily_return'].rolling(window=3*21).std()
        ohlc_data['6m_volatility'] = ohlc_data['daily_return'].rolling(window=6*21).std()
        ohlc_data['9m_volatility'] = ohlc_data['daily_return'].rolling(window=9*21).std()

        # Get the latest volatility values and the corresponding date
        latest_date = ohlc_data.index[-1]
        latest_3m_volatility = ohlc_data['3m_volatility'].iloc[-1]
        latest_6m_volatility = ohlc_data['6m_volatility'].iloc[-1]
        latest_9m_volatility = ohlc_data['9m_volatility'].iloc[-1]

        # Append results
        results.append({
            'Ticker': ticker, 
            'Date': latest_date, 
            '3 month volatility': latest_3m_volatility, 
            '6 month volatility': latest_6m_volatility, 
            '9 month volatility': latest_9m_volatility
        })

    return pd.DataFrame(results)

# Example usage
# ohlc_data = download_historical_data(tickers, start, end, interval)
volatility_df = calculate_volatility(historical_data)

def calculate_moving_average_returns(ohlc_data_dict):
    """
    Calculate 1-month, 3-month, 6-month, and 9-month moving averages of monthly returns for each stock in the ohlc_data_dict.

    :param ohlc_data_dict: Dictionary with tickers as keys and OHLC data as values.
    :return: DataFrame with 'Ticker', 'Date', '1 month MA returns', '3 month MA returns', '6 month MA returns', and '9 month MA returns'.
    """
    results = []

    for ticker, ohlc_data in ohlc_data_dict.items():
        # Ensure data is sorted by date
        ohlc_data = ohlc_data.sort_index()

        # Calculate monthly returns
        ohlc_data['monthly_return'] = ohlc_data['Adj Close'].pct_change()

        # Calculate moving averages for 1, 3, 6, and 9 months
        ohlc_data['1m_ma_returns'] = ohlc_data['monthly_return'].rolling(window=1).mean()
        ohlc_data['3m_ma_returns'] = ohlc_data['monthly_return'].rolling(window=3).mean()
        ohlc_data['6m_ma_returns'] = ohlc_data['monthly_return'].rolling(window=6).mean()
        ohlc_data['9m_ma_returns'] = ohlc_data['monthly_return'].rolling(window=9).mean()

        # Get the latest moving average values and the corresponding date
        latest_date = ohlc_data.index[-1]
        latest_1m_ma_returns = ohlc_data['1m_ma_returns'].iloc[-1]
        latest_3m_ma_returns = ohlc_data['3m_ma_returns'].iloc[-1]
        latest_6m_ma_returns = ohlc_data['6m_ma_returns'].iloc[-1]
        latest_9m_ma_returns = ohlc_data['9m_ma_returns'].iloc[-1]

        # Append results
        results.append({
            'Ticker': ticker, 
            'Date': latest_date, 
            '1 month MA returns': latest_1m_ma_returns,
            '3 month MA returns': latest_3m_ma_returns, 
            '6 month MA returns': latest_6m_ma_returns, 
            '9 month MA returns': latest_9m_ma_returns
        })

    return pd.DataFrame(results)

ma_returns_df = calculate_moving_average_returns(historical_data)

def fetch_latest_ohlc_data(ohlc_data_dict):
    """
    Fetch the latest OHLC data for each stock in the ohlc_data_dict.

    :param ohlc_data_dict: Dictionary with tickers as keys and OHLC data as values.
    :return: DataFrame with 'Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'.
    """
    results = []

    for ticker, ohlc_data in ohlc_data_dict.items():
        # Ensure data is sorted by date
        ohlc_data = ohlc_data.sort_index()

        # Get the latest OHLC data
        latest_data = ohlc_data.iloc[-1]
        latest_date = ohlc_data.index[-1]

        # Append results
        results.append({
            'Ticker': ticker,
            'Date': latest_date,
            'Open': latest_data['Open'],
            'High': latest_data['High'],
            'Low': latest_data['Low'],
            'Close': latest_data['Close'],
            'Volume': latest_data['Volume']
        })

    return pd.DataFrame(results)

# Example usage
# ohlc_data_dict = download_historical_data(tickers, start, end, interval)
latest_ohlc_df = fetch_latest_ohlc_data(historical_data)


def check_uptrends(ohlc_data, short_term_window=50, long_term_window=200):
    """
    Check if stocks are in an uptrend and return a DataFrame with details.

    :param ohlc_data: Dictionary with tickers as keys and OHLC data as values.
    :param short_term_window: Window size for short-term moving average.
    :param long_term_window: Window size for long-term moving average.
    :return: DataFrame with columns for Ticker, Date, uptrend_status, MA_short_term, and MA_long_term.
    """
    all_tickers_df = pd.DataFrame()

    for ticker, data in ohlc_data.items():
        data['MA_short_term'] = data['Close'].rolling(window=short_term_window).mean()
        data['MA_long_term'] = data['Close'].rolling(window=long_term_window).mean()
        data['uptrend_status'] = data['MA_short_term'] > data['MA_long_term']
        
        # Extracting the last row for each ticker
        latest_data = data.iloc[-1].copy()
        latest_data['Ticker'] = ticker
        latest_data['Date'] = data.index[-1]
        
        # Append to the combined DataFrame
        all_tickers_df = all_tickers_df.append(latest_data, ignore_index=True)

    return all_tickers_df[['Ticker', 'Date', 'uptrend_status', 'MA_short_term', 'MA_long_term']]

check_uptrends=check_uptrends(historical_data)

def calculate_latest_rsi(ohlc_data_dict, period=14):
    """
    Calculate the latest RSI for each stock in the ohlc_data_dict.

    :param ohlc_data_dict: Dictionary with tickers as keys and OHLC data as values.
    :param period: Period for calculating RSI (default is 14 days).
    :return: DataFrame with the latest RSI, ticker, and date for each stock.
    """
    results = []

    for ticker, ohlc_data in ohlc_data_dict.items():
        # Ensure data is sorted by date
        ohlc_data = ohlc_data.sort_index()

        # Calculate daily returns
        delta = ohlc_data['Close'].diff()

        # Make two series: one for gains and one for losses
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        # Calculate the average gain and loss
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # Calculate the relative strength (RS)
        rs = avg_gain / avg_loss

        # Calculate the RSI
        rsi = 100 - (100 / (1 + rs))

        # Get the latest RSI value and the corresponding date
        latest_rsi = rsi.iloc[-1]
        latest_date = ohlc_data.index[-1]

        # Append results
        results.append({'Ticker': ticker, 'Date': latest_date, 'RSI': latest_rsi})

    return pd.DataFrame(results)


latest_rsi_df = calculate_latest_rsi(historical_data)


def calculate_stochastic_oscillator(ohlc_data_dict, k_period=14, d_period=3):
    """
    Calculate the Stochastic Oscillator for each stock in the ohlc_data_dict.

    :param ohlc_data_dict: Dictionary with tickers as keys and OHLC data as values.
    :param k_period: Period for calculating %K (default is 14 days).
    :param d_period: Period for calculating %D, the moving average of %K (default is 3 days).
    :return: DataFrame with the latest %K, %D, ticker, and date for each stock.
    """
    results = []

    for ticker, ohlc_data in ohlc_data_dict.items():
        # Ensure data is sorted by date
        ohlc_data = ohlc_data.sort_index()

        # Calculate %K
        lowest_low = ohlc_data['Low'].rolling(window=k_period).min()
        highest_high = ohlc_data['High'].rolling(window=k_period).max()
        ohlc_data['%K'] = ((ohlc_data['Close'] - lowest_low) / (highest_high - lowest_low)) * 100

        # Calculate %D
        ohlc_data['%D'] = ohlc_data['%K'].rolling(window=d_period).mean()

        # Get the latest %K, %D values and the corresponding date
        latest_k = ohlc_data['%K'].iloc[-1]
        latest_d = ohlc_data['%D'].iloc[-1]
        latest_date = ohlc_data.index[-1]

        # Append results
        results.append({'Ticker': ticker, 'Date': latest_date, '%K': latest_k, '%D': latest_d})

    return pd.DataFrame(results)

"""A %K or %D value below 20 can indicate oversold conditions, suggesting a potential uptrend if the value starts rising above 20.
A %K or %D value above 80 can indicate overbought conditions, suggesting a potential downtrend if the value starts falling below 80.
Values between 20 and 80 typically indicate a sideways or consolidating market"""
stochastic_oscillator_df = calculate_stochastic_oscillator(historical_data)


def calculate_macd_and_trend(ohlc_data_dict):
    """
    Calculate the MACD and trend analysis for each stock.

    :param ohlc_data_dict: Dictionary with tickers as keys and OHLC data as values.
    :return: DataFrame with MACD, signal line, trend, ticker, and date for each stock.
    """
    results = []

    for ticker, ohlc_data in ohlc_data_dict.items():
        # Ensure data is sorted by date
        ohlc_data = ohlc_data.sort_index()

        # Calculate the short-term and long-term EMAs
        short_ema = ohlc_data['Close'].ewm(span=12, adjust=False).mean()
        long_ema = ohlc_data['Close'].ewm(span=26, adjust=False).mean()

        # Calculate the MACD line and the signal line
        ohlc_data['MACD'] = short_ema - long_ema
        ohlc_data['Signal_Line'] = ohlc_data['MACD'].ewm(span=9, adjust=False).mean()

        # Get the latest MACD and Signal Line values and the corresponding date
        latest_macd = ohlc_data['MACD'].iloc[-1]
        latest_signal = ohlc_data['Signal_Line'].iloc[-1]
        latest_date = ohlc_data.index[-1]

        # Determine the trend
        if latest_macd > latest_signal:
            trend = "Potential Uptrend"
        elif latest_macd < latest_signal:
            trend = "Potential Downtrend"
        else:
            trend = "Sideways/Consolidating"

        # Append results
        results.append({'Ticker': ticker, 'Date': latest_date, 'MACD': latest_macd, 'Signal_Line': latest_signal, 'Trend': trend})

    return pd.DataFrame(results)


macd_trend_df = calculate_macd_and_trend(historical_data)


def calculate_obv_and_trend(ohlc_data_dict):
    """
    Calculate the On-Balance Volume (OBV) and trend analysis for each stock.

    :param ohlc_data_dict: Dictionary with tickers as keys and OHLC data as values.
    :return: DataFrame with OBV, trend, ticker, and date for each stock.
    """
    results = []

    for ticker, ohlc_data in ohlc_data_dict.items():
        # Ensure data is sorted by date
        ohlc_data = ohlc_data.sort_index()

        # Initialize OBV column
        ohlc_data['OBV'] = 0

        # Calculate OBV
        for i in range(1, len(ohlc_data)):
            if ohlc_data['Close'][i] > ohlc_data['Close'][i-1]:
                ohlc_data['OBV'][i] = ohlc_data['OBV'][i-1] + ohlc_data['Volume'][i]
            elif ohlc_data['Close'][i] < ohlc_data['Close'][i-1]:
                ohlc_data['OBV'][i] = ohlc_data['OBV'][i-1] - ohlc_data['Volume'][i]
            else:
                ohlc_data['OBV'][i] = ohlc_data['OBV'][i-1]

        # Get the latest OBV value and the corresponding date
        latest_obv = ohlc_data['OBV'].iloc[-1]
        latest_date = ohlc_data.index[-1]

        # Determine the trend (basic analysis)
        trend = "Indeterminate"
        if len(ohlc_data) > 1:
            if ohlc_data['OBV'].iloc[-1] > ohlc_data['OBV'].iloc[-2]:
                trend = "Potential Uptrend"
            elif ohlc_data['OBV'].iloc[-1] < ohlc_data['OBV'].iloc[-2]:
                trend = "Potential Downtrend"

        # Append results
        results.append({'Ticker': ticker, 'Date': latest_date, 'OBV': latest_obv, 'Trend': trend})

    return pd.DataFrame(results)

obv_trend_df = calculate_obv_and_trend(historical_data)

import pandas as pd

def calculate_parabolic_sar_and_trend(ohlc_data_dict):
    """
    Calculate the Parabolic SAR and trend analysis for each stock.

    :param ohlc_data_dict: Dictionary with tickers as keys and OHLC data as values.
    :return: DataFrame with Parabolic SAR, trend, ticker, and date for each stock.
    """
    results = []

    for ticker, ohlc_data in ohlc_data_dict.items():
        # Ensure data is sorted by date
        ohlc_data = ohlc_data.sort_index()

        # Initialize Parabolic SAR columns
        ohlc_data['SAR'] = 0.0
        ohlc_data['EP'] = 0.0  # Extreme Point
        ohlc_data['AF'] = 0.02  # Acceleration Factor

        # Initial values
        ohlc_data['SAR'][0] = ohlc_data['Low'][0]
        ohlc_data['EP'][0] = ohlc_data['High'][0]
        uptrend = True

        for i in range(1, len(ohlc_data)):
            if uptrend:
                ohlc_data['SAR'][i] = ohlc_data['SAR'][i-1] + ohlc_data['AF'][i-1] * (ohlc_data['EP'][i-1] - ohlc_data['SAR'][i-1])
                if ohlc_data['High'][i] > ohlc_data['EP'][i-1]:
                    ohlc_data['EP'][i] = ohlc_data['High'][i]
                    ohlc_data['AF'][i] = min(ohlc_data['AF'][i-1] + 0.02, 0.20)
                else:
                    ohlc_data['EP'][i] = ohlc_data['EP'][i-1]
                    ohlc_data['AF'][i] = ohlc_data['AF'][i-1]

                if ohlc_data['Low'][i] < ohlc_data['SAR'][i]:
                    uptrend = False
                    ohlc_data['SAR'][i] = ohlc_data['EP'][i-1]
            else:
                ohlc_data['SAR'][i] = ohlc_data['SAR'][i-1] - ohlc_data['AF'][i-1] * (ohlc_data['SAR'][i-1] - ohlc_data['EP'][i-1])
                if ohlc_data['Low'][i] < ohlc_data['EP'][i-1]:
                    ohlc_data['EP'][i] = ohlc_data['Low'][i]
                    ohlc_data['AF'][i] = min(ohlc_data['AF'][i-1] + 0.02, 0.20)
                else:
                    ohlc_data['EP'][i] = ohlc_data['EP'][i-1]
                    ohlc_data['AF'][i] = ohlc_data['AF'][i-1]

                if ohlc_data['High'][i] > ohlc_data['SAR'][i]:
                    uptrend = True
                    ohlc_data['SAR'][i] = ohlc_data['EP'][i-1]

        # Get the latest SAR value and the corresponding date
        latest_sar = ohlc_data['SAR'].iloc[-1]
        latest_date = ohlc_data.index[-1]

        # Determine the trend
        trend = "Uptrend" if uptrend else "Downtrend"

        # Append results
        results.append({'Ticker': ticker, 'Date': latest_date, 'SAR': latest_sar, 'Trend': trend})

    return pd.DataFrame(results)



parabolic_sar_trend_df = calculate_parabolic_sar_and_trend(historical_data)



def rank_stocks_by_obv(ohlc_data_dict):
    """
    Rank stocks based on the change in On-Balance Volume (OBV).

    :param ohlc_data_dict: Dictionary with tickers as keys and OHLC data as values.
    :return: DataFrame with stocks ranked by OBV change.
    """
    obv_change = {}

    for ticker, data in ohlc_data_dict.items():
        # Calculate OBV
        obv = (data['Volume'] * (~data['Close'].diff().le(0) * 2 - 1)).cumsum()

        # Calculate the change in OBV
        obv_change[ticker] = obv.iloc[-1] - obv.iloc[0]

    # Create a DataFrame and rank stocks
    obv_rank_df = pd.DataFrame(list(obv_change.items()), columns=['Ticker', 'OBV_Change'])
    obv_rank_df.sort_values(by='OBV_Change', ascending=False, inplace=True)

    return obv_rank_df



obv_ranked_stocks = rank_stocks_by_obv(historical_data)



def ATR(DF,n=14):
     ### n is the number of days, commonly 14 or 21 for daily data 
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df['Adj Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Adj Close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2

# def slope(ser,n):
#     "function to calculate the slope of n consecutive points on a plot"
#     slopes = [i*0 for i in range(n-1)]
#     for i in range(n,len(ser)+1):
#         y = ser[i-n:i]
#         x = np.array(range(n))
#         y_scaled = (y - y.min())/(y.max() - y.min())
#         x_scaled = (x - x.min())/(x.max() - x.min())
#         x_scaled = sm.add_constant(x_scaled)
#         model = sm.OLS(y_scaled,x_scaled)
#         results = model.fit()
#         slopes.append(results.params[-1])
#     slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
#     return np.array(slope_angle)

def renko_DF(DF):
    "function to convert ohlc data into renko bricks"
    df = DF.copy()
    df.reset_index(inplace=True)
    df = df.iloc[:,[0,1,2,3,4,5]]
    df.columns = ["date","open","high","low","close","volume"]
    df2 = Renko(df)
    #df2.brick_size = max(0.5,round(ATR(DF,120)["ATR"][-1],0))
    df2.brick_size = max(0.5,round(ATR(DF, 14)["ATR"][-1], 0))    
    print(df2.brick_size)
    renko_df = df2.get_ohlc_data()
    renko_df["bar_num"] = np.where(renko_df["uptrend"]==True,1,np.where(renko_df["uptrend"]==False,-1,0))
    for i in range(1,len(renko_df["bar_num"])):
        if renko_df["bar_num"][i]>0 and renko_df["bar_num"][i-1]>0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
        elif renko_df["bar_num"][i]<0 and renko_df["bar_num"][i-1]<0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
    renko_df.drop_duplicates(subset="date",keep="last",inplace=True)
    return renko_df

def MACD(DF,a,b,c):
    """function to calculate MACD
       typical values a = 12; b =26, c =9"""
    df = DF.copy()
    df["MA_Fast"]=df["Adj Close"].ewm(span=a,min_periods=a).mean()
    df["MA_Slow"]=df["Adj Close"].ewm(span=b,min_periods=b).mean()
    df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
    df["Signal"]=df["MACD"].ewm(span=c,min_periods=c).mean()
    df.dropna(inplace=True)
    return (df["MACD"],df["Signal"])

def CAGR(DF):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    n = len(df)/(12)
    CAGR = (df["cum_return"].tolist()[-1])**(1/n) - 1
    return CAGR

def volatility(DF):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    vol = df["ret"].std() * np.sqrt(12)
    return vol

def sharpe(DF,rf):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df) - rf)/volatility(df)
    return sr
    

def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd

def OBV(DF):
    """function to calculate On Balance Volume"""
    df = DF.copy()
    df['daily_ret'] = df['Adj Close'].pct_change()
    df['direction'] = np.where(df['daily_ret']>=0,1,-1)
    df['direction'][0] = 0
    df['vol_adj'] = df['Volume'] * df['direction']
    df['obv'] = df['vol_adj'].cumsum()
    return df['obv']

def slope(ser, n):
    slopes = [np.nan] * len(ser)
    for i in range(n - 1, len(ser)):
        y = ser[i - n + 1:i + 1]
        if y.notna().all():  # Ensure no NaNs in the window
            x = np.array(range(n))
            y_scaled = (y - y.min()) / (y.max() - y.min())
            x_scaled = (x - x.min()) / (x.max() - x.min())
            x_scaled = sm.add_constant(x_scaled)
            model = sm.OLS(y_scaled, x_scaled)
            results = model.fit()
            slopes[i] = results.params[-1]
    slope_angle = np.rad2deg(np.arctan(np.array(slopes)))
    return np.array(slope_angle)


def generate_macd_renko_signals(ohlc_data, tickers):
    """
    Generate trading signals (Buy/Sell) based on Renko bar numbers and MACD for each ticker.

    :param ohlc_data: Dictionary with tickers as keys and OHLC data as values.
    :param tickers: List of ticker symbols.
    :return: DataFrame with 'Date', 'Ticker', 'Signal', 'bar_num', 'MACD', 'MACD_slope'.
    """
    ohlc_renko = {}
    results = []

    for ticker in tickers:
        print("Processing for", ticker)
        # Assuming renko_DF and MACD functions are defined
        renko = renko_DF(ohlc_data[ticker])
        renko.columns = ["Date", "open", "high", "low", "close", "uptrend", "bar_num"]

        merged_data = ohlc_data[ticker].merge(renko.loc[:, ["Date", "bar_num"]], how="outer", on="Date")
        merged_data["bar_num"].fillna(method='ffill', inplace=True)
        merged_data["macd"], merged_data["macd_sig"] = MACD(merged_data, 12, 26, 9)
        merged_data["macd_slope"] = slope(merged_data["macd"], 5)
        merged_data["macd_sig_slope"] = slope(merged_data["macd_sig"], 5)

        # Initialize signal
        signal = ""

        # Check the latest data for signal
        latest_data = merged_data.iloc[-1]
        if latest_data["bar_num"] >= 2 and latest_data["macd"] > latest_data["macd_sig"] and latest_data["macd_slope"] > latest_data["macd_sig_slope"]:
            signal = "Buy"
        elif latest_data["bar_num"] <= -2 and latest_data["macd"] < latest_data["macd_sig"] and latest_data["macd_slope"] < latest_data["macd_sig_slope"]:
            signal = "Sell"

        # Append results
        results.append({
            'Date': latest_data['Date'],
            'Ticker': ticker,
            'Signal': signal,
            'bar_num': latest_data['bar_num'],
            'MACD': latest_data['macd'],
            'MACD_slope': latest_data['macd_slope']
        })

    return pd.DataFrame(results)

signals_df = generate_macd_renko_signals(historical_data, tickers)



# def slope(ser,n):
#     "function to calculate the slope of n consecutive points on a plot"
#     slopes = [i*0 for i in range(n-1)]
#     for i in range(n,len(ser)+1):
#         y = ser[i-n:i]
#         x = np.array(range(n))
#         y_scaled = (y - y.min())/(y.max() - y.min())
#         x_scaled = (x - x.min())/(x.max() - x.min())
#         x_scaled = sm.add_constant(x_scaled)
#         model = sm.OLS(y_scaled,x_scaled)
#         results = model.fit()
#         slopes.append(results.params[-1])
#     slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
#     return np.array(slope_angle)


def slope(ser, n):
    slopes = [np.nan] * len(ser)
    for i in range(n - 1, len(ser)):
        y = ser[i - n + 1:i + 1]
        if y.notna().all():  # Ensure no NaNs in the window
            x = np.array(range(n))
            y_scaled = (y - y.min()) / (y.max() - y.min())
            x_scaled = (x - x.min()) / (x.max() - x.min())
            x_scaled = sm.add_constant(x_scaled)
            model = sm.OLS(y_scaled, x_scaled)
            results = model.fit()
            slopes[i] = results.params[-1]
    slope_angle = np.rad2deg(np.arctan(np.array(slopes)))
    return np.array(slope_angle)



def MACD(DF,a,b,c):
    """function to calculate MACD
       typical values a = 12; b =26, c =9"""
    df = DF.copy()
    df["MA_Fast"]=df["Adj Close"].ewm(span=a,min_periods=a).mean()
    df["MA_Slow"]=df["Adj Close"].ewm(span=b,min_periods=b).mean()
    df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
    df["Signal"]=df["MACD"].ewm(span=c,min_periods=c).mean()
    df.dropna(inplace=True)
    return (df["MACD"],df["Signal"])




def generate_macd_renko_signals(ohlc_data, tickers):
    """
    Generate trading signals (Buy/Sell) based on Renko bar numbers and MACD for each ticker.

    :param ohlc_data: Dictionary with tickers as keys and OHLC data as values.
    :param tickers: List of ticker symbols.
    :return: DataFrame with 'Date', 'Ticker', 'Signal', 'bar_num', 'MACD', 'MACD_slope'.
    """
    ohlc_renko = {}
    results = []

    for ticker in tickers:
        print("Processing for", ticker)
        # Assuming renko_DF and MACD functions are defined
        renko = renko_DF(ohlc_data[ticker])
        renko.columns = ["Date", "open", "high", "low", "close", "uptrend", "bar_num"]

        merged_data = ohlc_data[ticker].merge(renko.loc[:, ["Date", "bar_num"]], how="outer", on="Date")
        merged_data["bar_num"].fillna(method='ffill', inplace=True)
        merged_data["macd"], merged_data["macd_sig"] = MACD(merged_data, 12, 26, 9)
        merged_data["macd_slope"] = slope(merged_data["macd"], 5)
        merged_data["macd_sig_slope"] = slope(merged_data["macd_sig"], 5)

        # Initialize signal
        signal = ""

        # Check the latest data for signal
        latest_data = merged_data.iloc[-1]
        if latest_data["bar_num"] >= 2 and latest_data["macd"] > latest_data["macd_sig"] and latest_data["macd_slope"] > latest_data["macd_sig_slope"]:
            signal = "Buy"
        elif latest_data["bar_num"] <= -2 and latest_data["macd"] < latest_data["macd_sig"] and latest_data["macd_slope"] < latest_data["macd_sig_slope"]:
            signal = "Sell"

        # Append results
        results.append({
            'Date': latest_data['Date'],
            'Ticker': ticker,
            'Signal': signal,
            'bar_num': latest_data['bar_num'],
            'MACD': latest_data['macd'],
            'MACD_slope': latest_data['macd_slope']
        })

    return pd.DataFrame(results)

signals_df = generate_macd_renko_signals(historical_data, tickers)


signals=signals_df.drop(['bar_num'],axis=1)
signals_renko_MACD_OBV=signals.merge(latest_signals_df, on =['Date','Ticker'],suffixes=['_MACD','_OBV'])



def calculate_atr_and_trend(ohlc_data_dict, atr_period=14):
    """
    Calculate the ATR, trend analysis, and stop loss for each stock.

    :param ohlc_data_dict: Dictionary with tickers as keys and OHLC data as values.
    :param atr_period: The period over which to calculate the ATR.
    :return: DataFrame with ATR, trend, stop loss, stop_loss_10, ticker, and date for each stock.
    """
    results = []

    for ticker, ohlc_data in ohlc_data_dict.items():
        # Ensure data is sorted by date
        ohlc_data = ohlc_data.sort_index()

        # Calculate True Range
        high_low = ohlc_data['High'] - ohlc_data['Low']
        high_close = np.abs(ohlc_data['High'] - ohlc_data['Close'].shift())
        low_close = np.abs(ohlc_data['Low'] - ohlc_data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate ATR
        ohlc_data['ATR'] = true_range.rolling(window=atr_period).mean()

        # Get the latest ATR value and the corresponding date
        latest_atr = ohlc_data['ATR'].iloc[-1]
        latest_date = ohlc_data.index[-1]

        # Simple trend analysis
        trend = "Uptrend" if ohlc_data['Close'].iloc[-1] > ohlc_data['Close'].iloc[-atr_period] else "Downtrend"

        # Basic stop loss calculation
        stop_loss = ohlc_data['Close'].iloc[-1] - 2 * latest_atr

        # Stop loss based on 90% of the last adjusted close
        stop_loss_10 = 0.9 * ohlc_data['Adj Close'].iloc[-1]
        stop_loss_5 = 0.95 * ohlc_data['Adj Close'].iloc[-1]

        # Append results
        results.append({
            'Ticker': ticker, 
            'Date': latest_date, 
            'ATR': latest_atr, 
            'ATR_Trend': trend, 
            'Stop_Loss_ATR': stop_loss,
            'Stop_Loss_10': stop_loss_10,
            'Stop_loss_5':stop_loss_5
        })

    return pd.DataFrame(results)

# Example usage
# historical_data = download_historical_data(tickers, start, end, interval)
atr_trend_df = calculate_atr_and_trend(historical_data)




# Merging the DataFrames on 'Ticker' and 'Date'
merged_df = parabolic_sar_trend_df.merge(obv_trend_df, on=['Ticker', 'Date'], how='inner',suffixes=['_SAR','_OBV'])
merged_df = merged_df.merge(macd_trend_df, on=['Ticker', 'Date'], how='inner')
merged_df = merged_df.rename(columns={'Trend': 'Trend_MACD'})
merged_df = merged_df.merge(stochastic_oscillator_df, on=['Ticker', 'Date'], how='inner')
merged_df = merged_df.merge(latest_rsi_df, on=['Ticker', 'Date'], how='inner')
merged_df = merged_df.merge(latest_ohlc_df, on=['Ticker', 'Date'], how='inner')
merged_df = merged_df.merge(volatility_df, on=['Ticker', 'Date'], how='inner')
merged_df = merged_df.merge(momentum_df, on=['Ticker', 'Date'], how='inner')
merged_df = merged_df.merge(ma_returns_df, on=['Ticker', 'Date'], how='inner')
merged_df = merged_df.merge(signals_renko_MACD_OBV, on=['Ticker', 'Date'], how='inner')
merged_df = merged_df.merge(atr_trend_df, on=['Ticker', 'Date'], how='inner')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
merged_df=merged_df.sort_values(by='6 month MA returns',ascending=False)
merged_df
