from financial_database import FinancialDatabase
from datetime import date, datetime


class Basket:
    """Class definition of Basket"""

    database_name = r'sqlite:///C:\Users\gafza\PycharmProjects\AlgorithmicTradingStrategies\stock_db_v3'

    def __init__(self, tickers: {str, list}, currency: str = None, total_return: bool = False, dividend_tax: float = 0):
        self.tickers = tickers
        self.currency = currency
        self.total_return = total_return
        self.dividend_tax = dividend_tax

    def basket_prices(self, start_date: {date, datetime}=None, end_date: {date, datetime}=None):
        financial_database_handler = FinancialDatabase(self.database_name, False)
        if self.total_return:
            return financial_database_handler.get_close_price_df(self.tickers, start_date, end_date, self.currency)
        else:
            return financial_database_handler.get_total_return_df(self.tickers, start_date, end_date, self.dividend_tax,
                                                                  self.currency)

    # ------------------------------------------------------------------------------------------------------------------
    # getter and setter methods

    @property
    def dividend_tax(self):
        return self._dividend_tax

    @dividend_tax.setter
    def dividend_tax(self, dividend_tax: float):
        if dividend_tax < 0:
            raise ValueError('dividend_tax needs to be greater or equal to zero.')
        else:
            self._dividend_tax = dividend_tax

    def __repr__(self):
        return "<Basket(#tickers = {}, currency = {currency}, {total_return}{dividend_tax})>"\
            .format(len(self.tickers), currency=self.currency if self.currency else 'local',
                    total_return='total return' if self.total_return else 'price return',
                    dividend_tax=' with ' + str(round(self.dividend_tax*100, 2)) + '% dividend tax' if self.dividend_tax and self.total_return else '')


def main():
    tickers = ["SAND.ST", "HM-B.ST", "AAK.ST"]
    basket = Basket(tickers, None, True, 0.0975)
    print(basket)
    print(basket.basket_prices())

if __name__ == '__main__':
    main()

