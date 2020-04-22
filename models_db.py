from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, ForeignKey, Date
from sqlalchemy.orm import relationship


# ------------------------------------------------------------------------------------------------------------------------
# models_db.py
# Create tables, or models, with SQA by creating a Python class with the fields that define the table, then ask SQA to
# build those models in our database.
# ------------------------------------------------------------------------------------------------------------------------

Base = declarative_base()
# A base class stores a catalog of classes and mapped tables in the Declarative system.
# Once a base class is declared, any number of mapped classes can be defined in terms of it.


class Underlying(Base):
    __tablename__ = 'underlying'
    id = Column(Integer, primary_key=True)
    ticker = Column(String, unique=True)  # 'symbol'
    underlying_type = Column(String)  # 'quoteType'
    long_name = Column(String)  # 'longName'
    short_name = Column(String)  # 'shortName'
    sector = Column(String)  # 'sector'
    industry = Column(String)  # 'industry'
    country = Column(String)  # 'country'
    city = Column(String)  # 'city'
    address = Column(String)  # combining 'address1' and 'address2'
    currency = Column(String)  # 'currency'
    description = Column(String)  # 'longBusinessSummary'
    web_site = Column(String)  # 'website'
    has_dividend_history = Column(Boolean, default=False)
    first_ex_div_date = Column(Date, nullable=True, default=None)
    latest_observation_date = Column(DateTime, nullable=True)
    latest_observation_date_with_values = Column(DateTime, nullable=True)
    oldest_observation_date = Column(DateTime, nullable=True)
    open_prices = relationship('OpenPrice', back_populates='underlyings', cascade='all, delete, delete-orphan')
    high_prices = relationship('HighPrice', back_populates='underlyings', cascade='all, delete, delete-orphan')
    low_prices = relationship('LowPrice', back_populates='underlyings', cascade='all, delete, delete-orphan')
    close_prices = relationship('ClosePrice', back_populates='underlyings', cascade='all, delete, delete-orphan')
    volumes = relationship('Volume', back_populates='underlyings', cascade='all, delete, delete-orphan')
    dividends = relationship('Dividend', back_populates='underlyings', cascade='all, delete, delete-orphan')

    def __repr__(self):
        if self.underlying_type == 'EQUITY':
            equity_info = ' sector={}, industry={},'.format(self.short_name, self.industry)
        else:
            equity_info = ','
        return "<Underlying(ticker={}, underlying_type={}, short name={},{} country={}, " \
               "currency={})>".format(self.ticker, self.underlying_type, self.short_name, equity_info,
                                      self.country, self.currency)


class OpenPrice(Base):
    __tablename__ = 'open_price'
    __valuename__ = 'open_quote'
    id = Column(Integer, primary_key=True)
    obs_date = Column(DateTime)
    open_quote = Column(Float)
    comment = Column(String)
    data_source = Column(String)
    underlying_id = Column(Integer, ForeignKey('underlying.id'))
    underlyings = relationship('Underlying', back_populates='open_prices')

    def __repr__(self):
        return "<OpenPrice(observation date={}, quote={}, {})>".format(self.obs_date, self.open_quote, self.comment)


class HighPrice(Base):
    __tablename__ = 'high_price'
    __valuename__ = 'high_quote'
    id = Column(Integer, primary_key=True)
    obs_date = Column(DateTime)
    high_quote = Column(Float)
    comment = Column(String)
    data_source = Column(String)
    underlying_id = Column(Integer, ForeignKey('underlying.id'))
    underlyings = relationship('Underlying', back_populates='high_prices')

    def __repr__(self):
        return "<HighPrice(observation date={}, quote={}, {})>".format(self.obs_date, self.high_quote, self.comment)


class LowPrice(Base):
    __tablename__ = 'low_price'
    __valuename__ = 'low_quote'
    id = Column(Integer, primary_key=True)
    obs_date = Column(DateTime)
    low_quote = Column(Float)
    comment = Column(String)
    data_source = Column(String)
    underlying_id = Column(Integer, ForeignKey('underlying.id'))
    underlyings = relationship('Underlying', back_populates='low_prices')

    def __repr__(self):
        return "<LowPrice(observation date={}, quote={}, {})>".format(self.obs_date, self.low_quote, self.comment)


class ClosePrice(Base):
    __tablename__ = 'close_price'
    __valuename__ = 'close_quote'
    id = Column(Integer, primary_key=True)
    obs_date = Column(DateTime)
    close_quote = Column(Float)
    comment = Column(String)
    data_source = Column(String)
    underlying_id = Column(Integer, ForeignKey('underlying.id'))
    underlyings = relationship('Underlying', back_populates='close_prices')

    def __repr__(self):
        return "<ClosePrice(observation date={}, quote={}, {})>".format(self.obs_date, self.close_quote, self.comment)


class Volume(Base):
    __tablename__ = 'volume'
    __valuename__ = 'volume_quote'
    id = Column(Integer, primary_key=True)
    obs_date = Column(DateTime)
    volume_quote = Column(Float)
    comment = Column(String)
    data_source = Column(String)
    underlying_id = Column(Integer, ForeignKey('underlying.id'))
    underlyings = relationship('Underlying', back_populates='volumes')

    def __repr__(self):
        return "<Volume(observation date={}, quote={}, {})>".format(self.obs_date, self.volume_quote, self.comment)


class Dividend(Base):
    __tablename__ = 'dividend'
    __valuename__ = 'dividend_amount'
    id = Column(Integer, primary_key=True)
    ex_div_date = Column(DateTime)
    dividend_amount = Column(Float)
    comment = Column(String)
    data_source = Column(String)
    underlying_id = Column(Integer, ForeignKey('underlying.id'))
    underlyings = relationship('Underlying', back_populates='dividends')

    def __repr__(self):
        return "<Dividend(ex-dividend date={}, dividend={})>".format(self.ex_div_date, self.dividend_amount)


def main():
    print(Dividend.__tablename__)


if __name__ == '__main__':
    main()


