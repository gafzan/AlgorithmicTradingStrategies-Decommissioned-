"""
general_tools.py
"""
from itertools import zip_longest
from operator import itemgetter
from datetime import date

# ______________________________________________________________________________________________________________________
# Handling lists and dictionaries


def _grouper(iterable, n, fill_value=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fill_value)


def list_grouper(iterable, n, fill_value=None):
    g = list(_grouper(iterable, n, fill_value))
    try:
        g[-1] = [e for e in g[-1] if e is not None]
        return [list(tup) for tup in g]
    except IndexError:
        return [[]]


def extend_dict(dict_1: dict, dict_2: dict) -> dict:
    """Assumes that dic_1 and dic_2 are both dictionaries. Returns the merged/combined dictionary of the two
    dictionaries."""
    return {**dict_1, **dict_2}


def reverse_dict(dict_: dict) -> dict:
    """Assumes that dict_ is a dictionary. Returns a new dictionary where the keys and values have been reversed.
    old dictionary: {keys: values}
    new dictionary: {values: keys}"""
    return {value: key for key, value in dict_.items()}


def check_required_keys(required_keys: list, dictionary: dict):
    """
    Raises an error if any of the elements in the given list does not exists as a key in the given dictionary
    :param required_keys: list
    :param dictionary: dict
    :return: None
    """
    if any(req_key not in dictionary.keys() for req_key in required_keys):
        raise ValueError("'%s' are not specified" % "', '".join(set(required_keys).difference(dictionary.keys())))
    return


def get_values_from_key_list(dictionary: dict, key_list: list):
    """
    Returns a list of values based on each key in the given list
    :param dictionary: dict
    :param key_list: list of keys
    :return:
    """
    return list(itemgetter(key_list)(*dictionary))


def translate_value_key_dict(dictionary: dict, new_old_key_map: dict, old_new_value_per_old_key_map: dict):
    """
    Adjust the keys and values of the given dictionary according to the specified mappers
    :param dictionary: dict
    :param new_old_key_map: dict {new key: old key}
    :param old_new_value_per_old_key_map: dict {old key: {old value: new value}}
    :return: dict
    """
    # find the keys where the corresponding value needs to change and
    value_adj_keys = [key for key in old_new_value_per_old_key_map.keys() if key in dictionary.keys()]
    # change each value according to the mapper
    for value_adj_key in value_adj_keys:
        dictionary.update(
            {
                value_adj_key: old_new_value_per_old_key_map[value_adj_key].get(
                    dictionary[value_adj_key],
                    dictionary[value_adj_key]
                )
            }
        )
    # change each key according to the mapper
    return dictionary

# ______________________________________________________________________________________________________________________
# Handling strings

def string_is_number(s: str) -> bool:
    """Assumes s is a string. Returns boolean. If the string can be converted to a number then True, else false."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def capital_letter_no_blanks(input_: {str, list}) -> {str, list}:
    """Assumes input_ is a string or list. Returns a string with capital letters and blanks replaced by '_'."""
    if isinstance(input_, str):
        return input_.upper().replace(' ', '_')
    elif isinstance(input_, list):  # if list, adjust all elements recursively
        new_list = []
        for e in input_:
            if isinstance(e, str):
                new_list.append(capital_letter_no_blanks(e))
            else:
                new_list.append(e)
        return new_list
    else:
        raise TypeError("Can't change to capital letters and remove blanks for an object of type {}."
                        .format(type(input_)))


def capital_letter_no_blanks_list(variable_list: list) -> list:
    """Assumes that variable_list is a list. Returns a new list where all strings have been adjusted to have capital
    letters and blanks replaced by '_'."""
    new_list = []
    for e in variable_list:
        if isinstance(e, str):
            new_list.append(capital_letter_no_blanks(e))
        else:
            new_list.append(e)
    return new_list


def progression_bar(counter: int, goal: int) -> None:
    """Assumes counter and goal are int. Script prints a progression bar."""
    result = progression_bar_str(counter, goal)
    print(result)


def progression_bar_str(counter: int, goal: int) -> str:
    """Assumes counter and goal are int. Script prints a progression bar."""
    if counter > goal:
        raise ValueError("'counter' needs to be smaller or equal to 'goal'")
    if counter < 0 or goal <= 0:
        raise ValueError("'counter' can't be negative and 'goal' needs to be larger than zero.")
    progression_percentage_str = str(round(100 * counter / goal, 2)) + '%'
    length = 100
    steps = length / goal
    return_string = '[{}{}] {} ({}/{})'.format(int(counter * steps) * '*', int((goal - counter) * steps) * ' ', progression_percentage_str, counter, goal)
    return return_string


def user_picks_element_from_list(list_: list):
    """Assumes that list_ is a list. Script will print a list of all the elements and then ask user to pick one.
    Returns the chosen element."""
    if len(list_) == 0:
        raise ValueError('List is empty.')
    for i in range(len(list_)):
        print('{}: {}'.format(i + 1, list_[i]))
    ask_user = True
    while ask_user:
        try:
            list_index = int(input('Enter a number between 1 and {}:'.format(len(list_))))
            assert 1 <= list_index <= len(list_)
            ask_user = False
        except (ValueError, AssertionError):
            pass
    return list_[list_index - 1]


def ask_user_yes_or_no(question: str)->bool:
    """
    Asks a question to user and user needs to say 'yes' or 'no' (several versions are accepted)
    :param question: str
    :return: bool
    """
    accpetable_yes = ['sure', 'yeah', 'yes', 'y']
    accpetable_no = ['no', 'n', 'nope']

    while True:
        answer = input(question + '\nYes or No?: ').lower()
        if answer in accpetable_yes:
            return True
        elif answer in accpetable_no:
            return False
        else:
            print("'{}' is not an acceptable answer...\n".format(answer))


def time_period_logger_msg(start_date: {date, None}, end_date: {date, None}):
    """
    Returns a string to be used in a logger message telling us about the time eriod we are looking at
    :param start_date: date, None
    :param end_date: date, None
    :return: str
    """
    return '' if start_date is None else ' from {}'.format(start_date) + '' if end_date is None else ' up to {}'.format(end_date)

