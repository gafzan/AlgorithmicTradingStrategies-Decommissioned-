from itertools import zip_longest

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
    if counter > goal:
        raise ValueError("'counter' needs to be smaller or equal to 'goal'")
    if counter < 0 or goal <= 0:
        raise ValueError("'counter' can't be negative and 'goal' needs to be larger than zero.")
    progression_percentage_str = str(round(100 * counter / goal, 2)) + '%'
    length = 100
    steps = length / goal
    print('[' + int(counter * steps) * '*' + int((goal - counter) * steps) * ' ' + ']  ' + progression_percentage_str
          + ' (' + str(counter) + ' / ' + str(goal) + ')')

