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
- **Signal**: *on a quarterly basis pick the 10 least volatile stocks*
- **Weight**: *use equal weighting i.e. 10% weight to each low volatility stock*

### Managing the data - Tutorial (1/3)
First we are going to use the GUI to add the data that we need. Later we will go though how to implement the same operation with a script. The *tickers* that we need can be found on [Yahoo! Finance](https://finance.yahoo.com/quote/%5EOMX/components?p=%5EOMX). Make sure you extract the rows from the "Symbol" column. 

Run the script *financial_database_gui.py* and select **Action:** *Add underlying*, **Method:** *Manually*, **Data source:** *Yahoo Finance* and press "Perform action".
Then paste the tickers you want to add to the data base as a string where each ticker is seperated by a comma (,) and press ENTER. 

<!--
Insert screenshot of GUI
-->
We could also add the data by executing the below script:
```
from database.config_database import my_database_name
from database.financial_database import YahooFinanceFeeder

# tickers of the stocks inlcuded in OMX 30 as of Sep 2020
tickers = ["ERIC-B.ST", "SEB-A.ST", "TEL2-B.ST", "SECU-B.ST", "ASSA-B.ST", "KINV-B.ST", "AZN.ST", "SKA-B.ST", "NDA-SE.ST", "ALFA.ST", "SKF-B.ST", "HM-B.ST", "ABB.ST", "SWED-A.ST", "SAND.ST", "BOL.ST", "SSAB-A.ST", "INVE-B.ST", "ATCO-A.ST", "TELIA.ST", "VOLV-B.ST", "SWMA.ST", "ATCO-B.ST", "ESSITY-B.ST", "HEXA-B.ST", "SHB-A.ST", "ELUX-B.ST", "SCA-B.ST", "GETI-B.ST", "ALIV-SDB.ST"]

# reference to a YahooFinanceFeeder object that is used to insert data using the Yahoo! Finance API
yf_feeder = YahooFinanceFeeder(my_database_name)

# add the OMX 30 stocks to your data base
yf_feeder.add_underlying(tickers)
```

### Setting up the strategy - Tutorial (2/3)
Now when we have the data, we can implement our strategy. We begin by importing what we need.
```
import pandas as pd

# data base modules
from database.config_database import my_database_name
from database.financial_database import YahooFinanceFeeder

# strategy modules
from algorithmic_strategy.investment_universe import InvestmentUniverse
from algorithmic_strategy.strategy_signal import VolatilityRankSignal
from algorithmic_strategy.strategy_weight import EqualWeight
from algorithmic_strategy.strategy import Index
```
We utilize the [```pandas``` package](https://pandas.pydata.org/pandas-docs/version/0.15/tutorials.html) to generate a rebalance calendar for the strategy.

To setup our investment universe we need to initialize an InvestmentUniverse object. In order to do that, we need to have a list of tickers and observation dates.

Licensed under [Apache License 2.0](LICENSE)







