# Execution
NSE['ticker'] = NSE['Symbol'] + '.NS'
tickers = NSE[0:150]['ticker']
results = analyze_portfolio_performance(tickers, m=10, x=10, rf=0.09, training_interval='1d',test_interval='1mo')

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

## Monthly returns
print(results['y'].groupby(by='Date')['capped_values'].mean())

### Annual return 

data = {
    'Date': results['test_return_df'][7:].index.to_list(),
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
