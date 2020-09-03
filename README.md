# Algorithmic Trading Strategies
***Version 1.0.0***

This project aims to be a toolbox for testing and analysing algorithmic trading strategies. At the moment, it is not meant
to be an automatic trading tool but rather be an enviroment where you can efficiently back test systematic trading strategies. 
The signals (e.g. low volatility stocks and simple moving average crosses) and weighting schemes available (e.g. equal weight) will be expanded over time.

The project has three main parts: **financial database**, **strategy** and **performance analysis**.

### 1. Financial database 
Handle underlying financial data such as OHLC prices, volumes and dividends in a database using SQLAlchemy ORM. Available data sources are so far Yahoo! Finance (using the [yfinance package](https://github.com/ranaroussi/yfinance)), Bloomberg and Excel. Using a GUI the user can add, delete, refresh and download financial data. 


### 2. Strategy
Setting up a strategy by first choosing your *investment universe* (e.g. Swedish stocks), perform *initial filters* to find underlyings eligible for investment (e.g. only stocks in a certain sector with enough liquidity), a *signal* that decides your portfolio selection (e.g. lowest volotile stocks) and then finaly picking a weighting mechanism for your strategy (e.g. equal weighting). 

A back test can then be performed using the financial database that can incorporate among other things transaction costs, fees, various rebalancing calendars and volatility target.

### 3. Performance analysis
The strategy can be analysed using various return and risk metrics such as rolling average return & volatility, maximum drawdown and active return compared to a choosen benchmark. The results can be saved in excel for further analysis in a standardized format.

---
## Tutorial
Now we are going to perform a simple (and not profitable) strategy that will be used to illustrate each of the parts of the project.

Our strategy will be the following:
- **Investment universe**: *OMX 30 (stocks of the 30 largest  firms traded on the Stockholm Stock Exchange)*
- **Eligibility filter**:
  1. *Liquidity > SEK 10 million*
  2. *Exclude stocks that are in the financial sector*
- **Signal**: *pick the 10 least volatile stocks*
- **Weight**: *use equal weighting i.e. 10% weight to each low volatility stock*

### Managing the data - Tutorial (1/3)
First we are going to use the GUI to add the data that we need. Later we will go though how to implement the same operation with a script. The *tickers* that we need can be found on [Yahoo! Finance](https://finance.yahoo.com/quote/%5EOMX/components?p=%5EOMX). Make sure you extract the rows from the "Symbol" column. 


### Setting up the strategy - Tutorial (1/3)

<!--
## Tutorial part 1 - Managing the data


## Tutorial part 2 - Simple (and terrible) trading strategy

## Tutorial part 3 - Performance evaluation

## License & copyright
-->
Licensed under [Apache License 2.0](LICENSE)







