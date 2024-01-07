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


## Download NSE data based on Market Cap
NSE=pd.read_excel('/Users/alekhsaxena/Downloads/MCAP31032023_0.xlsx')

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



## Function to calculate monthly returns 
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




def calculate_3_month_momentum(ohlc_data):
    """
    Calculate 3-month momentum for each stock.

    :param ohlc_data: Dictionary with tickers as keys and OHLC data as values.
    :return: DataFrame with 3-month momentum for each ticker.
    """
    ohlc_dict = copy.deepcopy(ohlc_data)
    momentum_df = pd.DataFrame()

    for ticker in ohlc_dict.keys():
        print(f"Calculating 3-month momentum for {ticker}")
        # Assuming approximately 21 trading days in a month
        ohlc_dict[ticker]['3m_momentum'] = ohlc_dict[ticker]['Adj Close'].pct_change(periods=3)
        momentum_df[ticker] = ohlc_dict[ticker]['3m_momentum']

    return momentum_df


def calculate_6_month_momentum(ohlc_data):
    """
    Calculate 3-month momentum for each stock.

    :param ohlc_data: Dictionary with tickers as keys and OHLC data as values.
    :return: DataFrame with 3-month momentum for each ticker.
    """
    ohlc_dict = copy.deepcopy(ohlc_data)
    momentum_df = pd.DataFrame()

    for ticker in ohlc_dict.keys():
        print(f"Calculating 3-month momentum for {ticker}")
        # Assuming approximately 21 trading days in a month
        ohlc_dict[ticker]['3m_momentum'] = ohlc_dict[ticker]['Adj Close'].pct_change(periods=6)
        momentum_df[ticker] = ohlc_dict[ticker]['3m_momentum']

    return momentum_df



def calculate_12_month_momentum(ohlc_data):
    """
    Calculate 3-month momentum for each stock.

    :param ohlc_data: Dictionary with tickers as keys and OHLC data as values.
    :return: DataFrame with 3-month momentum for each ticker.
    """
    ohlc_dict = copy.deepcopy(ohlc_data)
    momentum_df = pd.DataFrame()

    for ticker in ohlc_dict.keys():
        print(f"Calculating 12-month momentum for {ticker}")
        # Assuming approximately 21 trading days in a month
        ohlc_dict[ticker]['3m_momentum'] = ohlc_dict[ticker]['Adj Close'].pct_change(periods=12)
        momentum_df[ticker] = ohlc_dict[ticker]['3m_momentum']

    return momentum_df


def calculate_3_month_moving_average_returns(ohlc_data):
    """
    Calculate 3-month moving averages of monthly returns for each stock.

    :param ohlc_data: Dictionary with tickers as keys and OHLC data as values.
    :return: DataFrame with 3-month moving averages of returns for each ticker.
    """
    ohlc_dict = copy.deepcopy(ohlc_data)
    ma_return_df = pd.DataFrame()

    for ticker in ohlc_dict.keys():
        # Calculate monthly return
        ohlc_dict[ticker]['monthly_return'] = ohlc_dict[ticker]['Adj Close'].pct_change(periods=1)  # Monthly return

        # Calculate 3-month moving average of monthly returns
        ma_column_name = '3m_ma_return'
        ohlc_dict[ticker][ma_column_name] = ohlc_dict[ticker]['monthly_return'].rolling(window=3).mean()
        ma_return_df[ticker] = ohlc_dict[ticker][ma_column_name]

    return ma_return_df



def calculate_6_month_moving_average_returns_daily(ohlc_data):
    """
    Calculate 6-month moving averages of monthly returns for each stock.

    :param ohlc_data: Dictionary with tickers as keys and OHLC data as values.
    :return: DataFrame with 6-month moving averages of returns for each ticker.
    """
    ohlc_dict = copy.deepcopy(ohlc_data)
    ma_return_df = pd.DataFrame()

    for ticker in ohlc_dict.keys():
        # Calculate monthly return
        ohlc_dict[ticker]['monthly_return'] = ohlc_dict[ticker]['Adj Close'].pct_change(periods=1*21)  # Monthly return

        # Calculate 6-month moving average of monthly returns
        ma_column_name = '6m_ma_return'
        ohlc_dict[ticker][ma_column_name] = ohlc_dict[ticker]['monthly_return'].rolling(window=6*21).mean()
        ma_return_df[ticker] = ohlc_dict[ticker][ma_column_name]

    return ma_return_df



def calculate_6_month_moving_average_returns_monthly(ohlc_data):
    """
    Calculate 6-month moving averages of monthly returns for each stock.

    :param ohlc_data: Dictionary with tickers as keys and OHLC data as values.
    :return: DataFrame with 6-month moving averages of returns for each ticker.
    """
    ohlc_dict = copy.deepcopy(ohlc_data)
    ma_return_df = pd.DataFrame()

    for ticker in ohlc_dict.keys():
        # Calculate monthly return
        ohlc_dict[ticker]['monthly_return'] = ohlc_dict[ticker]['Adj Close'].pct_change(periods=1)  # Monthly return

        # Calculate 6-month moving average of monthly returns
        ma_column_name = '6m_ma_return'
        ohlc_dict[ticker][ma_column_name] = ohlc_dict[ticker]['monthly_return'].rolling(window=6).mean()
        ma_return_df[ticker] = ohlc_dict[ticker][ma_column_name]

    return ma_return_df


def calculate_3_month_ema_returns(ohlc_data):
    """
    Calculate 3-month exponential moving averages of monthly returns for each stock.

    :param ohlc_data: Dictionary with tickers as keys and OHLC data as values.
    :return: DataFrame with 3-month EMAs of returns for each ticker.
    """
    ohlc_dict = copy.deepcopy(ohlc_data)
    ema_return_df = pd.DataFrame()

    for ticker in ohlc_dict.keys():
        # Calculate monthly return
        ohlc_dict[ticker]['monthly_return'] = ohlc_dict[ticker]['Adj Close'].pct_change(periods=1)  # Monthly return

        # Calculate 3-month EMA of monthly returns
        ema_column_name = '3m_ema_return'
        span =  3  # Span for 3-month EMA
        ohlc_dict[ticker][ema_column_name] = ohlc_dict[ticker]['monthly_return'].ewm(span=span, adjust=False).mean()
        ema_return_df[ticker] = ohlc_dict[ticker][ema_column_name]

    return ema_return_df



def calculate_6_month_ema_returns(ohlc_data):
    """
    Calculate 3-month exponential moving averages of monthly returns for each stock.

    :param ohlc_data: Dictionary with tickers as keys and OHLC data as values.
    :return: DataFrame with 3-month EMAs of returns for each ticker.
    """
    ohlc_dict = copy.deepcopy(ohlc_data)
    ema_return_df = pd.DataFrame()

    for ticker in ohlc_dict.keys():
        # Calculate monthly return
        ohlc_dict[ticker]['monthly_return'] = ohlc_dict[ticker]['Adj Close'].pct_change(periods=1)  # Monthly return

        # Calculate 6-month EMA of monthly returns
        ema_column_name = '6m_ema_return'
        span =  6  # Span for 6-month EMA
        ohlc_dict[ticker][ema_column_name] = ohlc_dict[ticker]['monthly_return'].ewm(span=span, adjust=False).mean()
        ema_return_df[ticker] = ohlc_dict[ticker][ema_column_name]

    return ema_return_df

def pflio(DF,m,x):
    """Returns cumulative portfolio return
    DF = dataframe with monthly return info for all stocks
    m = number of stock in the portfolio
    x = number of underperforming stocks to be removed from portfolio monthly"""
    df = DF.copy()
    portfolio = []
    monthly_portfolio=[]
    monthly_ret = [0]
    for i in range(len(df)):
        if len(portfolio) > 0:
            monthly_ret.append(df[portfolio].iloc[i,:].mean())
            bad_stocks = df[portfolio].iloc[i,:].sort_values(ascending=True)[:x].index.values.tolist()
            portfolio = [t for t in portfolio if t not in bad_stocks]
        fill = m - len(portfolio)
        new_picks = df.iloc[i,:].sort_values(ascending=False)[:fill].index.values.tolist()
        portfolio = portfolio + new_picks
        print(portfolio)
        monthly_portfolio.append(portfolio)
    monthly_ret_df = pd.DataFrame(np.array(monthly_ret),columns=["mon_ret"])
    return (monthly_ret_df,monthly_portfolio)

def calculate_portfolio_monthly_returns(return_df, portfolio):
    """
    Calculate monthly returns for a given portfolio.

    :param return_df: DataFrame with monthly returns for each stock.
    :param portfolio: List of stocks included in the portfolio.
    :return: DataFrame with calculated monthly returns for the portfolio.
    """
    monthly_return = []
    back = return_df.shift(-1, axis=0)
    back = back.reset_index()
    y=pd.DataFrame()
    print(back)

    for i in range(len(back) - 1):
        m=[]
        
        m.append(back.loc[i, 'Date'])
        port = portfolio[i]  # Assuming portfolio[0] is a list of tickers for the current month
        x=back.loc[i, port]
        i=x[port].T
        i=i.reset_index()
        i.columns=['stocks','Returns']
        i['Date']=m*len(i.index)
        y=pd.concat([i,y])
        actual_return = x.mean()
        monthly_return.append(actual_return)

    monthly_return_df = pd.DataFrame(np.array(monthly_return), columns=["mon_ret"])
    return monthly_return_df,y


def process_financial_data(portfolio, train_return_df, test_return_df):
    # Assuming calculate_monthly_returns and pflio are defined elsewhere
    df = pd.DataFrame([portfolio])
    df = df.T
    df.columns = ['Stocks']
    df['Date'] = train_return_df.index

    test_return_df = test_return_df.reset_index()
    test_return_df['Mapped Date'] = test_return_df['Date'].apply(lambda x: df[df['Date'] < x]['Date'].max())

    merged_df_corrected = test_return_df.merge(df, left_on='Mapped Date', right_on='Date', suffixes=('_test', '_portfolio'))

    y = pd.DataFrame()
    monthly_return = []
    for i in range(merged_df_corrected.shape[0]):
        m = []
        m.append(merged_df_corrected.loc[i, 'Date_test'])
        port = merged_df_corrected.loc[i, 'Stocks']
        actual_return = merged_df_corrected.loc[i, port].mean()
        monthly_return.append(actual_return)
        x = merged_df_corrected.loc[i, port]
        i = x.T
        i = i.reset_index()
        i.columns = ['stocks', 'Returns']
        i['Date'] = m * len(i.index)
        y = pd.concat([i, y])

    monthly_return_df = pd.DataFrame(np.array(monthly_return), columns=["mon_ret"])

    return monthly_return_df, y



def slope(ser, n):
    # Function to calculate the slope of n consecutive points on a plot
    slopes = [i*0 for i in range(n-1)]
    for i in range(n, len(ser)+1):
        y = ser[i-n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min()) / (y.max() - y.min())
        x_scaled = (x - x.min()) / (x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled, x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)

def MACD(df, a, b, c):
    # Function to calculate MACD
    df["MA_Fast"] = df["Adj Close"].ewm(span=a, min_periods=a).mean()
    df["MA_Slow"] = df["Adj Close"].ewm(span=b, min_periods=b).mean()
    df["MACD"] = df["MA_Fast"] - df["MA_Slow"]
    df["Signal"] = df["MACD"].ewm(span=c, min_periods=c).mean()
    df.dropna(inplace=True)
    return df["MACD"], df["Signal"]



def calculate_macd_slope(ohlc_data):
    """
    Calculate the slope of the MACD for each stock.

    :param ohlc_data: Dictionary with tickers as keys and OHLC data as values.
    :return: DataFrame with the slope of the MACD for each ticker.
    """
    ohlc_dict = copy.deepcopy(ohlc_data)
    macd_slope_df = pd.DataFrame()

    for ticker in ohlc_dict.keys():
        # Calculate MACD
        macd, macd_signal = MACD(ohlc_dict[ticker], 12, 26, 9)

        # Calculate the slope of the MACD
        macd_slope = slope(macd, 5)

        # Check if the DataFrame is not empty
        if not ohlc_dict[ticker].empty and len(macd_slope) > 0:
            # Reindex macd_slope to match ohlc_dict[ticker]'s index
            macd_slope_reindexed = pd.Series(macd_slope, index=ohlc_dict[ticker].index[-len(macd_slope):])
            ohlc_dict[ticker]['MACD_slope'] = macd_slope_reindexed
            macd_slope_df[ticker] = macd_slope_reindexed
        else:
            # Handle empty DataFrame or no data case
            ohlc_dict[ticker]['MACD_slope'] = pd.Series(dtype='float64')
            macd_slope_df[ticker] = pd.Series(dtype='float64')

    return macd_slope_df




def analyze_portfolio_performance(tickers, m, x, rf=0.09, training_interval='1d',test_interval='1mo'):
    """
    Analyze portfolio performance based on tickers.

    :param tickers: List containing ticker symbols.
    :param m: Number of stocks in the portfolio.
    :param x: Number of underperforming stocks to be removed from the portfolio monthly.
    :param rf: Risk-free rate for Sharpe ratio calculation.
    :param interval: Data interval for historical data download.
    :return: Tuple containing CAGR, Sharpe ratio, and maximum drawdown.
    """
    if training_interval=='1d':
        p=252
    elif training_interval=='1wk':
        p=52
    else:
        p=12 ## Monthly
        
    if test_interval=='1d':
        q=252
    elif test_interval=='1wk':
        q=52
    else:
        q=12 ## Monthly
    print(p)
    print(q)
    
    start_date = dt.datetime.today() - dt.timedelta(365*5)
    end_date = dt.datetime.today()
    ohlc_train = download_historical_data(tickers, start_date, end_date, interval=training_interval)
    ohlc_test = download_historical_data(tickers, start_date, end_date, interval=test_interval)
    
    #train_return_df=calculate_monthly_returns(ohlc_train)

    ## Changing Strategy from monthly return to 3months momentum
    #train_return_df=calculate_3_month_momentum(ohlc_train)

    ## Changing Strategy from monthly return to 6months momentum
    #train_return_df=calculate_6_month_momentum(ohlc_train)
    
    ## Changing Strategy from monthly return to 12months momentum
    #train_return_df=calculate_12_month_momentum(ohlc_train)  
    
    ## Changing Strategy from monthly return to 3months Moving Average of returns
    #train_return_df=calculate_3_month_moving_average_returns(ohlc_train)  
    
    ## Changing Strategy from monthly return to 3months Moving Average of returns
    if p==12:
        train_return_df=calculate_6_month_moving_average_returns_monthly(ohlc_train) 
    else :
        train_return_df=calculate_6_month_moving_average_returns_daily(ohlc_train) 
        
    
    
    
    
    ## Changing Strategy from monthly return to 3months Moving Average of returns
    #train_return_df=calculate_macd_slope(ohlc_train) 
    
    train_return_df=train_return_df.dropna(axis=0, how='all')
    train_return_df=train_return_df.dropna(axis=1, how='any')
    
    
    
    test_return_df = calculate_monthly_returns(ohlc_test)
    test_return_df = test_return_df.iloc[1:].dropna(axis=1, how='any')
    
    print(test_return_df.shape)
    train_return_df=train_return_df.reset_index()
    test_return_df=test_return_df.reset_index()
    train_return_df['Mapped Date2'] = train_return_df['Date'].apply(lambda x: test_return_df[test_return_df['Date'] > x]['Date'].min())
    train_return_df=train_return_df.loc[train_return_df.groupby('Mapped Date2')['Date'].idxmax()]
    print(train_return_df.shape)
    train_return_df=train_return_df.drop('Mapped Date2',axis=1)
    train_return_df=train_return_df.set_index('Date')
    test_return_df=test_return_df.set_index('Date')
    print(train_return_df.shape)





    portfolio_return_df, portfolio = pflio(train_return_df, m, x)
    print(portfolio_return_df.shape)
    cagr_value = CAGR(portfolio_return_df,f=p)
    sharpe_ratio = sharpe(portfolio_return_df, rf,f=p)
    max_drawdown_value = max_dd(portfolio_return_df)
    test_return_df=test_return_df.dropna(axis=0, how='all')
    
    if p==12:
        monthly_return_df, y = process_financial_data(portfolio, train_return_df, test_return_df)
    else:
        monthly_return_df, y = process_financial_data_daily(portfolio, train_return_df, test_return_df)
        
        

    
    #monthly_return_df,y = calculate_portfolio_monthly_returns(test_return_df, portfolio)
    y['capped_values']=y['Returns'].clip(lower=-0.2, upper=0.2)
    actual_cagr_value = CAGR(monthly_return_df,f=q)
    actual_sharpe_ratio = sharpe(monthly_return_df, rf,f=q)
    actual_max_drawdown_value = max_dd(monthly_return_df)
    capped_returns=pd.DataFrame(np.array(y['capped_values']),columns=["mon_ret"])
    capped_cagr_value=CAGR(capped_returns,f=q)
    capped_sharpe_ratio = sharpe(capped_returns, rf,f=q)
    capped_max_drawdown_value = max_dd(capped_returns)

    return {
        'cagr': cagr_value,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown_value,
        'actual_cagr': actual_cagr_value,
        'actual_sharpe_ratio': actual_sharpe_ratio,
        'actual_max_drawdown': actual_max_drawdown_value,
        'portfolio': portfolio,
        'monthly_return_df':monthly_return_df,
        'test_return_df':test_return_df,
        'y':y,
        'capped_cagr_value':capped_cagr_value,
        'capped_sharpe_ratio':capped_sharpe_ratio,
        'capped_max_drawdown_value':capped_max_drawdown_value,
        'train_return_df':train_return_df
        }


print('cagr:',results['cagr'])
print('sharpe_ratio:',results['sharpe_ratio'])
print('max_drawdown_value:',results['max_drawdown'])
print('actual_cagr',results['actual_cagr'])
print('actual_sharpe_ratio',results['actual_sharpe_ratio'])
print('actual_max_drawdown',results['actual_max_drawdown'])
print('capped_cagr',results['capped_cagr_value'])
print('capped_sharpe_ratio',results['capped_sharpe_ratio'])
print('capped_max_drawdown',results['capped_max_drawdown_value'])
print('monthly_return_df',results['monthly_return_df'])
print('portfolio',results['portfolio'])


data = {
    'Date': results['test_return_df'][6:].index.to_list(),
    #'Monthly_Return': results['monthly_return_df']['mon_ret']# Sample data
    'Monthly_Return': results['y'].groupby(by='Date')['capped_values'].mean().tolist()# Sample data
}

df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# Function to calculate annual returns from monthly returns
def calculate_annual_returns(df, return_col):
    annual_returns = {}
    for year in range(df.index.year.min(), df.index.year.max() + 1):
        yearly_data = df[df.index.year == year]
        annual_return = (yearly_data[return_col] + 1).prod() - 1
        annual_returns[year] = annual_return
    return annual_returns

# Calculate annual returns
annual_returns = calculate_annual_returns(df, 'Monthly_Return')

# Convert the annual returns to a DataFrame for better visualization
annual_returns_df = pd.DataFrame(list(annual_returns.items()), columns=['Year', 'Annual_Return'])
print(annual_returns_df)
