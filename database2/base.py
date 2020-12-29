"""
base.py
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

db_name = 'my_db_file'
file_location = r'C:\Users\gafza\PycharmProjects\AlgorithmicTradingStrategies\database2'
sql_echo = False

# 'declarative_base' is a factory function that defines a base class. We will inherit ths class definition when we
# define our other tables so SQLALchemy can interpret them correctly (Base will contain information that we need)
Base = declarative_base()

# 'create_engine' builds a factory of database connections. It works as a registry which provide connectivity to a
# particular database server (in this case 'SQLite')
# the engine handles Pools (re-using of pre-initialized objects to enhance performance) and Dialects (able to
# communicate with the different database engines) When using the ORM, we typically don’t use the Engine directly once
# created; instead, it’s used behind the scenes by the ORM. It will not 'connect' here since that is done later when it
# is necessary
engine = create_engine(r'sqlite:///' + file_location + '\\' + db_name + '.db', echo=sql_echo)

# create a "session factory" (a session is created each time Session is called) and tell the factory to always interact
# with the given database connection i.e. the engine
_SessionFactory = sessionmaker(bind=engine)


def session_factory():
    """
    Sets up the tables and returns an instance of a session from the session factory that will be used to
    communicate with the database
    :return: Session
    """
    # migrate the changes we have made to the initial table models by referring to the engine
    # 'metadata' describes the structure of the database (tables, columns, rows etc.) in terms of python data structures
    # we tell the metadata to emit 'CREATE TABLE' statements to all the defined tables to the given engine and commit
    Base.metadata.create_all(engine)
    return _SessionFactory()
