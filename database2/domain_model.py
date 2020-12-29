"""
domain_model.py
"""
from datetime import date
import re
import inspect  # used to find all the argument names in the __init__ of Instrument class definition

from sqlalchemy import Column, ForeignKey, func
from sqlalchemy import Integer, String, Boolean, Date, Float
from sqlalchemy.orm import synonym, relationship
from sqlalchemy.ext.declarative import declared_attr

from database2.base import Base


available_data_sources = ['YAHOO', 'BLOOMBERG', 'BORSDATA', 'EXCEL']
available_instrument_types = ['STOCK', 'INDEX', 'FUTURE', 'ETF', 'BOND', 'FX']


class Instrument(Base):
    __tablename__ = 'instruments'

    id = Column(Integer, primary_key=True)
    _ticker = Column('ticker', String, unique=True)
    _data_source = Column('data_source', String, nullable=False)
    _instrument_type = Column('instrument_type', String, nullable=False)
    _asset_class = Column('asset_class', String, nullable=False)  # e.g. commodity
    _sub_asset_class = Column('sub_asset_class', String, nullable=True)  # e.g. gold
    _currency = Column('currency', String, nullable=False)
    _exchange = Column('exchange', String, nullable=False)
    _short_name = Column('short_name', String, nullable=False)
    long_name = Column('long_name', String, nullable=True)
    _sector = Column('sector', String, nullable=True)
    _industry = Column('industry', String, nullable=True)
    country = Column('country', String, nullable=True)
    city = Column('city', String, nullable=True)
    address = Column('address', String, nullable=True)
    description = Column('description', String, nullable=True)
    website = Column('website', String, nullable=True)
    has_dividend_history = Column('has_dividend_history', Boolean)

    # dates
    first_ex_div_date = Column('first_ex_div_date', Date, nullable=True)
    # latest_observation_date = Column('latest_observation_date', Date, nullable=True)
    # oldest_observation_date = Column('oldest_observation_date', Date, nullable=True)
    expiry_date = Column('expiry_date', Date, nullable=True)
    last_open_price_refresh = Column(Date, nullable=True)
    last_high_price_refresh = Column(Date, nullable=True)
    last_low_price_refresh = Column(Date, nullable=True)
    last_close_price_refresh = Column(Date, nullable=True)
    last_volume_refresh = Column(Date, nullable=True)
    last_dividend_refresh = Column(Date, nullable=True)
    last_split_refresh = Column(Date, nullable=True)

    # relationships
    open_prices = relationship('OpenPrice', cascade='all, delete, delete-orphan')
    high_prices = relationship('HighPrice', cascade='all, delete, delete-orphan')
    low_prices = relationship('LowPrice', cascade='all, delete, delete-orphan')
    close_prices = relationship('ClosePrice', cascade='all, delete, delete-orphan')
    volumes = relationship('Volume', cascade='all, delete, delete-orphan')
    dividends = relationship('Dividend', cascade='all, delete, delete-orphan')
    splits = relationship('Split', cascade='all, delete, delete-orphan')

    def __init__(self, ticker: str, data_source: str, instrument_type: str, asset_class: str, currency: str,
                 short_name: str, long_name: str = None, exchange: str = None, sub_asset_class: str = None, sector: str = None, industry: str = None, country: str = None,
                 city: str = None, address: str = None, description: str = None, website: str = None, has_dividend_history: bool = None,
                 expiry_date: date = None):
        self.ticker = ticker
        self.data_source = data_source
        self.instrument_type = instrument_type
        self.asset_class = asset_class
        self.sub_asset_class = sub_asset_class
        self.currency = currency
        self.short_name = short_name
        self.long_name = long_name
        self.exchange = exchange
        self.sector = sector
        self.industry = industry
        self.country = country
        self.city = city
        self.address = address
        self.description = description
        self.website = website
        self.has_dividend_history = False if has_dividend_history is None else has_dividend_history
        self.expiry_date = expiry_date

    def __repr__(self):
        return f"<{Instrument.__name__}(ticker={self.ticker}, data_source={self.data_source}, instrument_type={self.instrument_type}, " \
            f"asset_class={self.asset_class}, currency={self.currency}, short_name={self.short_name})>"

    # getter and setter methods
    # ticker (capital letters)
    @property
    def ticker(self):
        return self._ticker

    @ticker.setter
    def ticker(self, ticker: str):
        self._ticker = ticker.upper()

    ticker = synonym('_ticker', descriptor=ticker)

    # data_source (capital letters and value needs to be in a specified list)
    @property
    def data_source(self):
        return self._data_source

    @data_source.setter
    def data_source(self, data_source: str):
        data_source = data_source.upper()
        if data_source in available_data_sources:
            self._data_source = data_source
        else:
            raise ValueError("'data_source' needs to be specified as one of '%s'" % "', '".join(available_data_sources))

    data_source = synonym('_data_source', descriptor=data_source)

    # instrument_type (capital letters and value needs to be in a specified list)
    @property
    def instrument_type(self):
        return self._instrument_type

    @instrument_type.setter
    def instrument_type(self, instrument_type: str):
        instrument_type = instrument_type.upper()
        if instrument_type in available_instrument_types:
            self._instrument_type = instrument_type
        else:
            raise ValueError("'instrument_type' needs to be specified as one of '%s'" % "', '".join(available_instrument_types))

    instrument_type = synonym('_instrument_type', descriptor=instrument_type)

    # asset_class (capital letters)
    @property
    def asset_class(self):
        return self._asset_class

    @asset_class.setter
    def asset_class(self, asset_class: str):
        self._asset_class = asset_class.upper()

    asset_class = synonym('_asset_class', descriptor=asset_class)

    # sub_asset_class (capital letters)
    @property
    def sub_asset_class(self):
        return self._sub_asset_class

    @sub_asset_class.setter
    def sub_asset_class(self, sub_asset_class: str):
        self._sub_asset_class = sub_asset_class.upper() if sub_asset_class else sub_asset_class

    sub_asset_class = synonym('_sub_asset_class', descriptor=sub_asset_class)

    # currency (capital letters)
    @property
    def currency(self):
        return self._currency

    @currency.setter
    def currency(self, currency: str):
        self._currency = currency.upper()

    currency = synonym('_currency', descriptor=currency)

    # short_name (capital letters)
    @property
    def short_name(self):
        return self._short_name

    @short_name.setter
    def short_name(self, short_name: str):
        self._short_name = short_name.upper()

    short_name = synonym('_short_name', descriptor=short_name)

    # exchange (capital letters)
    @property
    def exchange(self):
        return self._exchange

    @exchange.setter
    def exchange(self, exchange: str):
        self._exchange = exchange.upper() if exchange else exchange

    exchange = synonym('_exchange', descriptor=exchange)

    # sector (capital letters and _ instead of blanks)
    @property
    def sector(self):
        return self._sector

    @sector.setter
    def sector(self, sector: str):
        self._sector = sector.upper().replace(' ', '_') if sector else sector

    sector = synonym('_sector', descriptor=sector)

    # industry (capital letters and _ instead of blanks)
    @property
    def industry(self):
        return self._industry

    @industry.setter
    def industry(self, industry: str):
        self._industry = industry.upper().replace(' ', '_') if industry else industry

    industry = synonym('_industry', descriptor=industry)


class _DataTableMixin:

    @declared_attr
    def __tablename__(self):
        # splits 'FirstSecond' to ['first', 'second']
        words = [word.lower() for word in re.findall('[A-Z][^A-Z]*', self.__name__)]
        return "%s" % "_".join(words) + 's'

    id = Column(Integer, primary_key=True)

    as_of_date = Column(Date)
    value = Column(Float)
    _source = Column('source', String)
    created_at = Column(Date, default=func.date(date.today()))

    # relationships
    @declared_attr
    def instrument_id(self):
        return Column(Integer, ForeignKey('instruments.id'))

    @declared_attr
    def refresh_info_column_name(self):
        words = [word.lower() for word in re.findall('[A-Z][^A-Z]*', self.__name__)]
        col_name = "last_{}_refresh".format("%s" % "_".join(words))
        if col_name in Instrument.__table__.c.keys():
            return col_name
        else:
            raise ValueError("Instrument table needs to have a column with the name '{}'\n{} = Column(Date, nullable=True)".format(col_name, col_name))

    def __init__(self, as_of_date: date, value: float, source: str):
        self.as_of_date = as_of_date
        self.value = value
        self.source = source

    def __repr__(self):
        return "<{}(as_of_date={}, value={}, source={}, created_at={})>".format(type(self).__name__, self.as_of_date,
                                                                                self.value, self.source, self.created_at)

    # capital letters and value needs to be in a specified list
    def get_source(self):
        return self._source

    def set_source(self, source):
        source = source.upper()
        if source in available_data_sources:
            self._source = source
        else:
            raise ValueError(
                "'source' needs to be specified as one of '%s'" % "', '".join(available_data_sources))

    @declared_attr
    def source(self):
        return synonym('_source', descriptor=property(self.get_source, self.set_source))


class OpenPrice(_DataTableMixin, Base):
    pass


class HighPrice(_DataTableMixin, Base):
    pass


class LowPrice(_DataTableMixin, Base):
    pass


class ClosePrice(_DataTableMixin, Base):
    pass


class Volume(_DataTableMixin, Base):
    pass


class Dividend(_DataTableMixin, Base):
    pass


class Split(_DataTableMixin, Base):
    pass


# this will be used by the financial database manager to check which data tables to refresh for each instrument
data_table_instrument_types_mapper = {OpenPrice: available_instrument_types,
                                      LowPrice: available_instrument_types,
                                      HighPrice: available_instrument_types,
                                      ClosePrice: available_instrument_types,
                                      Volume: [instrument_type for instrument_type in available_instrument_types if instrument_type != 'INDEX'],
                                      Dividend: [instrument_type for instrument_type in available_instrument_types if instrument_type not in ['INDEX', 'FX', 'BOND']],
                                      Split: [instrument_type for instrument_type in available_instrument_types if instrument_type not in ['INDEX', 'FX', 'BOND']]}

init_instrument_arg_names = [arg_name for arg_name in list(inspect.signature(Instrument).parameters.keys())
                             if arg_name not in ['ticker', 'data_source']]

instrument_default_value_map = {key: (None if Instrument.__table__.c[key].nullable else 'N/A') for key in init_instrument_arg_names}



# print(inspect.signature(Instrument.__init__).__dict__)
# init_instrument_arg_names = list(dict(inspect.signature(Instrument.__init__).keys())[1:]
# print(init_instrument_arg_names)
#
# test
# from database2.base import session_factory
#
# session = session_factory()
# inst = Instrument(ticker='MCD', data_source='bloomberg', instrument_type='stock', asset_class='you ass', currency='usd', short_name='swag', exchange='STO')
#
# inst.close_prices.extend([ClosePrice(date.today(), 69, source='YAhoo')])
# inst.volumes.append(Volume(date.today(), 700, 'bloomberg'))
#
# session.add(inst)
# session.commit()

#
# table_list = [ClosePrice, Volume]
#
# q = session.query(*table_list).filter(
#     *[d_tab.instrument_id.in_([1]) for d_tab in table_list]
# ).all()
# print(q)
#
# session.commit()
# session.close()

# class OpenPrice(Base):
#     __tablename__ = 'open_prices'
#
#     id = Column(Integer, primary_key=True)
#     date = Column(Date)
#     price = Column(Float)
#     _data_source = Column('data_source', String)
#     comment = Column(String)
#
#     refresh_info_column_name = 'last_open_price_refresh'
#     if refresh_info_column_name not in Instrument.__dict__:
#         raise ValueError("value of 'refresh_info_column_name' attribute needs to be a column name in the instruments table")
#
#     # relationships
#     instrument_id = Column(Integer, ForeignKey('instruments.id'))
#
#     def __init__(self, date: _date, price: float, data_source: str, comment: str = None):
#         self.date = date
#         self.price = price
#         self.data_source = data_source
#         self.comment = comment
#
#     def __repr__(self):
#         return f"{OpenPrice.__name__}(date={self.date}, price={self.price}, data_source={self.data_source})>"
#
#     # capital letters and value needs to be in a specified list
#     @property
#     def data_source(self):
#         return self._data_source
#
#     @data_source.setter
#     def data_source(self, data_source: str):
#         data_source = data_source.upper()
#         if data_source in available_data_sources:
#             self._data_source = data_source
#         else:
#             raise ValueError(
#                 "'data_source' needs to be specified as one of '%s'" % "', '".join(available_data_sources))
#
#     data_source = synonym('_data_source', descriptor=data_source)
#
#
# class HighPrice(Base):
#     __tablename__ = 'high_prices'
#
#     id = Column(Integer, primary_key=True)
#     date = Column(Date)
#     price = Column(Float)
#     _data_source = Column('data_source', String)
#     comment = Column(String)
#
#     refresh_info_column_name = 'last_high_price_refresh'
#     if refresh_info_column_name not in Instrument.__dict__:
#         raise ValueError("value of 'refresh_info_column_name' attribute needs to be a column name in the instruments table")
#
#     # relationships
#     instrument_id = Column(Integer, ForeignKey('instruments.id'))
#
#     def __init__(self, date: _date, price: float, data_source: str, comment: str = None):
#         self.date = date
#         self.price = price
#         self.data_source = data_source
#         self.comment = comment
#
#     def __repr__(self):
#         return f"{HighPrice.__name__}(date={self.date}, price={self.price}, data_source={self.data_source})>"
#
#     # capital letters and value needs to be in a specified list
#     @property
#     def data_source(self):
#         return self._data_source
#
#     @data_source.setter
#     def data_source(self, data_source: str):
#         data_source = data_source.upper()
#         if data_source in available_data_sources:
#             self._data_source = data_source
#         else:
#             raise ValueError(
#                 "'data_source' needs to be specified as one of '%s'" % "', '".join(available_data_sources))
#
#     data_source = synonym('_data_source', descriptor=data_source)
#
#
# class LowPrice(Base):
#     __tablename__ = 'low_prices'
#
#     id = Column(Integer, primary_key=True)
#     date = Column(Date)
#     price = Column(Float)
#     _data_source = Column('data_source', String)
#     comment = Column(String)
#
#     refresh_info_column_name = 'last_low_price_refresh'
#     if refresh_info_column_name not in Instrument.__dict__:
#         raise ValueError("value of 'refresh_info_column_name' attribute needs to be a column name in the instruments table")
#
#     # relationships
#     instrument_id = Column(Integer, ForeignKey('instruments.id'))
#
#     def __init__(self, date: _date, price: float, data_source: str, comment: str = None):
#         self.date = date
#         self.price = price
#         self.data_source = data_source
#         self.comment = comment
#
#     def __repr__(self):
#         return f"{LowPrice.__name__}(date={self.date}, price={self.price}, data_source={self.data_source})>"
#
#     # capital letters and value needs to be in a specified list
#     @property
#     def data_source(self):
#         return self._data_source
#
#     @data_source.setter
#     def data_source(self, data_source: str):
#         data_source = data_source.upper()
#         if data_source in available_data_sources:
#             self._data_source = data_source
#         else:
#             raise ValueError(
#                 "'data_source' needs to be specified as one of '%s'" % "', '".join(available_data_sources))
#
#     data_source = synonym('_data_source', descriptor=data_source)
#
#
# class ClosePrice(Base):
#     __tablename__ = 'close_prices'
#
#     id = Column(Integer, primary_key=True)
#     date = Column(Date)
#     price = Column(Float)
#     _data_source = Column('data_source', String)
#     comment = Column(String)
#
#     refresh_info_column_name = 'last_close_price_refresh'
#     if refresh_info_column_name not in Instrument.__dict__:
#         raise ValueError("value of 'refresh_info_column_name' attribute needs to be a column name in the instruments table")
#
#     # relationships
#     instrument_id = Column(Integer, ForeignKey('instruments.id'))
#
#     def __init__(self, date: _date, price: float, data_source: str, comment: str = None):
#         self.date = date
#         self.price = price
#         self.data_source = data_source
#         self.comment = comment
#
#     def __repr__(self):
#         return f"{ClosePrice.__name__}(date={self.date}, price={self.price}, data_source={self.data_source})>"
#
#     # capital letters and value needs to be in a specified list
#     @property
#     def data_source(self):
#         return self._data_source
#
#     @data_source.setter
#     def data_source(self, data_source: str):
#         data_source = data_source.upper()
#         if data_source in available_data_sources:
#             self._data_source = data_source
#         else:
#             raise ValueError(
#                 "'data_source' needs to be specified as one of '%s'" % "', '".join(available_data_sources))
#
#     data_source = synonym('_data_source', descriptor=data_source)
#
#
# class Volume(Base):
#     __tablename__ = 'volumes'
#
#     id = Column(Integer, primary_key=True)
#     date = Column(Date)
#     volume = Column(Float)
#     _data_source = Column('data_source', String)
#     comment = Column(String)
#
#     refresh_info_column_name = 'last_close_price_refresh'
#     if refresh_info_column_name not in Instrument.__dict__:
#         raise ValueError("value of 'refresh_info_column_name' attribute needs to be a column name in the instruments table")
#
#     # relationships
#     instrument_id = Column(Integer, ForeignKey('instruments.id'))
#
#     def __init__(self, date: _date, volume: float, data_source: str, comment: str = None):
#         self.date = date
#         self.volume = volume
#         self.data_source = data_source
#         self.comment = comment
#
#     def __repr__(self):
#         return f"{Volume.__name__}(date={self.date}, volume={self.volume}, data_source={self.data_source})>"
#
#     # capital letters and value needs to be in a specified list
#     @property
#     def data_source(self):
#         return self._data_source
#
#     @data_source.setter
#     def data_source(self, data_source: str):
#         data_source = data_source.upper()
#         if data_source in available_data_sources:
#             self._data_source = data_source
#         else:
#             raise ValueError(
#                 "'data_source' needs to be specified as one of '%s'" % "', '".join(available_data_sources))
#
#     data_source = synonym('_data_source', descriptor=data_source)
#
#
# class Dividend(Base):
#     __tablename__ = 'dividends'
#
#     id = Column(Integer, primary_key=True)
#     date = Column('date (ex-dividend date)', Date)
#     dividend_amount = Column(Float)
#     _data_source = Column('data_source', String)
#     comment = Column(String)
#
#     refresh_info_column_name = 'last_dividend_refresh'
#     if refresh_info_column_name not in Instrument.__dict__:
#         raise ValueError("value of 'refresh_info_column_name' attribute needs to be a column name in the instruments table")
#
#     # relationships
#     instrument_id = Column(Integer, ForeignKey('instruments.id'))
#
#     def __init__(self, date: _date, dividend_amount: float, data_source: str, comment: str = None):
#         self.date = date
#         self.dividend_amount = dividend_amount
#         self.data_source = data_source
#         self.comment = comment
#
#     def __repr__(self):
#         return f"{Dividend.__name__}(date={self.date}, dividend_amount={self.dividend_amount}, data_source={self.data_source})>"
#
#     # capital letters and value needs to be in a specified list
#     @property
#     def data_source(self):
#         return self._data_source
#
#     @data_source.setter
#     def data_source(self, data_source: str):
#         data_source = data_source.upper()
#         if data_source in available_data_sources:
#             self._data_source = data_source
#         else:
#             raise ValueError(
#                 "'data_source' needs to be specified as one of '%s'" % "', '".join(available_data_sources))
#
#     data_source = synonym('_data_source', descriptor=data_source)




# test
# from database2.base import session_factory
#
# session = session_factory()

# instrument1 = Instrument(ticker='mcd', data_source='excel', instrument_type='stock', asset_class='equity',
#                          currency='usd', short_name='mcdonalds', sector='food and stuff')
#
# instrument2 = Instrument(ticker='gs', data_source='bloomberg', instrument_type='stock', asset_class='equity',
#                          currency='usd', short_name='mcdonalds', sector='fraud')
#
# instrument1.close_prices = [ClosePrice(date(2020, 1, 1), 42, 'bloomberg'), ClosePrice(date(2020, 1, 2), 88, 'bloomberg')]
# instrument2.close_prices = [ClosePrice(date.today(), 69, 'bloomberg')]
# session.add(instrument1)
# session.add(instrument2)

# # print all instruments
# instruments = session.query(Instrument).all()
# print(instruments)
# for inst in instruments:
#     print(f'{inst.ticker}({inst.data_source}) is in the {inst.sector} sector and {inst.industry} industry')

# session.add(Instrument(ticker='SPY', data_source='YAHOO', instrument_type='ETF', asset_class='equity', currency='usd', short_name='s&p 500 etf'))

# session.delete(instruments[1])
# instruments[1].open_prices.extend([OpenPrice(date(1999, 1, 1), 100, 'yahoo'), OpenPrice(date(1998, 1, 2), 200, 'yahoo')])

# instruments[2].dividends.extend([Dividend(date(1992, 9, 12), 1.4, 'borsdata'), Dividend(date(1992, 9, 13), 1.3, 'borsdata')])
# instruments[1].dividends.append(Dividend(date(1984, 4, 4), 4, 'bloomberg'))

#
# session.commit()
# session.close()











