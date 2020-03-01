# Algorithmic Trading Strategies
**Version 1.0.0**

This project aims to be a toolbox for testing and analysing algorithmic trading strategies. At the moment, it is not meant
to be an automatic trading tool but rather be an enviroment where you can efficiently back test systematic trading strategies. 
The signals (e.g. low volatility stocks and simple moving average crosses) and weighting schemes available (e.g. equal weight) will be expanded over time.

The project has three main parts: *financial database*, *strategy back test* and *performance analysis*.

### 1. Financial database 
Handling underlying financial data such as OHLC prices, volumes and dividends in a database using SQLAlchemy ORM. Able to create or delete rows in the database. 

**Modules: models_db.py, financial_database.py**

### 2. Strategy back test
Setting up a strategy by first choosing your investment universe from the databsase, perform initial filters 
such as liquidity, chosing a signal that decides your portfolio selection, picking a weighting mechanism for your strategy (e.g. equal weighting) and then finaly back test your strategy.

**Modules: index_weight.py, index_signal.py, index.py**

### 3. Performance analysis
Display informative plots of your results and saving it to an excel file for more thorough analysis.

**Modules: excel_tools.py, financial_tools.py**


