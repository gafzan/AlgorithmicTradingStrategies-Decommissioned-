
my_database_name = r'sqlite:///[ENTER FULL PATH]'
if '[ENTER FULL PATH]' in my_database_name:
    raise ValueError('Name your database file.')
