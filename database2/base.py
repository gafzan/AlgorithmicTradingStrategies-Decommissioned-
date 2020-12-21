"""
base.py
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

db_name = 'my_db_file'
file_location = r'C:\Users\gafza\PycharmProjects\AlgorithmicTradingStrategies\database2'
sql_echo = False


# parent class that will be inherited when creating our table class definitions
Base = declarative_base()

# create an engine
# the engine handles Pools (re-using of pre-initialized objects to enhance performance) and Dialects (able to
# communicate with the different database engines) When using the ORM, we typically don’t use the Engine directly once
# created; instead, it’s used behind the scenes by the ORM.
engine = create_engine(r'sqlite:///' + file_location + '\\' + db_name + '.db', echo=sql_echo)

# create a "session factory" (a session is created each time Session is called)
_SessionFactory = sessionmaker(bind=engine)


def session_factory():
    """
    Sets up the tables and returns an instance of a session from the session factory that will be used to
    communicate with the database
    :return: Session
    """
    Base.metadata.create_all(engine)
    return _SessionFactory()
