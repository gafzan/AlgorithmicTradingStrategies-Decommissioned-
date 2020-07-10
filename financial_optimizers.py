"""
financial_optimizers.py
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

# my modules
from general_tools import progression_bar_str


def total_allocation_constraint(weight, allocation: float, upper_bound: bool = True):
    """
    Used for inequality constraint for the total allocation.
    :param weight: np.array
    :param allocation: float
    :param upper_bound: bool if true the constraint is from above (sum of weights <= allocation) else from below
    (sum of weights <= allocation)
    :return: np.array
    """
    if upper_bound:
        return allocation - weight.sum()
    else:
        return weight.sum() - allocation


def portfolio_variance(weights: np.array, covariance_matrix: np.array):
    """
    Return the portfolio variance
    :param weights: np.array
    :param covariance_matrix: np.array
    :return: float
    """
    return np.matmul(np.matmul(weights, covariance_matrix), np.transpose(weights))


def portfolio_volatility(weights: np.array, covariance_matrix: np.array):
    """
    Return the portfolio volatility
    :param weights: np.array
    :param covariance_matrix: np.array
    :return: float
    """
    port_variance = portfolio_variance(weights, covariance_matrix)
    return np.sqrt(port_variance)


def negative_risk_aversion_adjusted_return(weight: np.array, covariance_matrix: np.array, mean_returns: np.array, lambda_: float):
    """
    Returns portfolio returns - lambda * portfolio volatility. This is minimized in a mean variance optimization framework.
    :param weight: np.array
    :param covariance_matrix: np.array
    :param mean_returns: np.array
    :param lambda_: float
    :return: float
    """
    port_ret = np.matmul(weight, np.transpose(mean_returns))
    port_vol = portfolio_volatility(weight, covariance_matrix)
    return lambda_ * port_vol - port_ret


def target_return_constraint(weight: np.array, mean_returns: np.array, target_return: float):
    """
    Used for inequality constraint for the target return.
    :param weight: np.array
    :param mean_returns: np.array
    :param target_return: float
    :return: float
    """
    port_ret = np.matmul(weight, np.transpose(mean_returns))
    return port_ret - target_return


def get_instrument_weight_bounds(min_instrument_weight: {float, list, tuple}, max_instrument_weight: {float, list, tuple}, num_assets: int):
    """
    Return a list of tuples containg the minimum and maximum allowed weights for each instrument.
    :param min_instrument_weight: {float, list, tuple}
    :param max_instrument_weight:  {float, list, tuple}
    :param num_assets: int
    :return: list
    """
    # setup the constraints for the minimum instrument weights
    try:
        if len(min_instrument_weight) != num_assets:
            raise ValueError(
                'Number of minimum instrument weights ({}) is not the same as number of assets ({}).'.format(
                    len(min_instrument_weight), num_assets))
        instrument_weight_bounds_min = [(min_i_w,) for min_i_w in min_instrument_weight]
    except TypeError:
        instrument_weight_bounds_min = num_assets * [(min_instrument_weight,)]

    # setup the constraints for the maximum instrument weights
    instrument_weight_bounds = []
    for i in range(len(instrument_weight_bounds_min)):
        try:
            if len(max_instrument_weight) != num_assets:
                raise ValueError(
                    'Number of maximum instrument weights ({}) is not the same as number of assets ({}).'.format(
                        len(max_instrument_weight), num_assets))
            min_max_tpl = instrument_weight_bounds_min[i] + (max_instrument_weight[i],)
        except TypeError:
            min_max_tpl = instrument_weight_bounds_min[i] + (max_instrument_weight,)
        instrument_weight_bounds.append(min_max_tpl)
    return instrument_weight_bounds


def optimal_mean_variance_portfolio_weights_with_constraints(mean_returns: np.array, covariance_matrix, initial_guess: np.array = None, min_total_weight: float = 0.,  max_total_weight: float = 1.,
                                                             min_instrument_weight: {float, list, tuple}=0., max_instrument_weight: {float, list, tuple}=1., risk_aversion_factor: float = 1, return_target: float = 0):
    num_assets = covariance_matrix.shape[0]
    if initial_guess is None:  # if not initial guess is provided, use equal weighting as an initial guess
        initial_guess = np.array(num_assets * [1 / num_assets])

    # returns a list of tuples [(min_weight_1, max_weight_1), (min_weight_2, max_weight_2), ...
    instrument_weight_bounds = get_instrument_weight_bounds(min_instrument_weight, max_instrument_weight, num_assets)

    # sum of all instrument weights should be within a certain region
    max_total_weight_cons = {'type': 'ineq',
                             'fun': total_allocation_constraint,
                             'args': (max_total_weight,)}
    min_total_weight_cons = {'type': 'ineq',
                             'fun': total_allocation_constraint,
                             'args': (min_total_weight, False, )}

    # the return target needs to be satisfied
    return_target_cons = {'type': 'ineq',
                          'fun': target_return_constraint,
                          'args': (mean_returns, return_target, )}
    total_weight_cons = [min_total_weight_cons, max_total_weight_cons, return_target_cons]

    # find the solution using an active set optimizer
    sol = minimize(negative_risk_aversion_adjusted_return, initial_guess, method='SLSQP', args=(covariance_matrix, mean_returns, risk_aversion_factor, ),
                   bounds=instrument_weight_bounds, constraints=total_weight_cons)
    return sol.x


def minimum_variance_portfolio_weights_with_constraints(covariance_matrix, initial_guess: np.array = None, max_total_weight: float = 1.,
                                                        min_instrument_weight: {float, list, tuple}=0., max_instrument_weight: {float, list, tuple}=1.):

    num_assets = covariance_matrix.shape[0]
    if initial_guess is None:  # if not initial guess is provided, use equal weighting as an initial guess
        initial_guess = np.array(num_assets * [1 / num_assets])

    # returns a list of tuples [(min_weight_1, max_weight_1), (min_weight_2, max_weight_2), ...
    instrument_weight_bounds = get_instrument_weight_bounds(min_instrument_weight, max_instrument_weight, num_assets)

    # sum of all instrument weights should sum up to a given amount
    max_total_weight_cons = {'type': 'eq',
                             'fun': total_allocation_constraint,
                             'args': (max_total_weight, )}

    # find the solution using an active set optimizer
    sol = minimize(portfolio_variance, initial_guess, method='SLSQP', args=(covariance_matrix, ),
                   bounds=instrument_weight_bounds, constraints=max_total_weight_cons)
    return sol.x


def minimum_variance_optimizer(covariance_matrix, initial_guess):
    pass


def _theoretical_mean_variance_optimizer(mean_returns: {np.array, None}, covariance_matrix: np.array, return_target: {float, None},
                                         allocation: float = 1.0)->np.array:
    """
    Finds the theoretical optimal weights of the portfolio within an optimal mean-variance framework. If no mean returns
    are provided, the optimizer returns the weights of the minimum variance portfolio. If no return target is provided,
    the optimizer solves for the highest of the given mean returns.
    The only constraint that the sum of all weights should be equal to a given allocation.
    This is done by solving the linear equations derived from setting the gradient of the Lagrangian to the zero vector.
    :param mean_returns: array
    :param covariance_matrix: array
    :param return_target: float
    :param allocation: float
    :return: np.array
    """
    # solve the linear equation derived from setting the gradient of the Lagrangian to the zero vector
    # set up the problem as A_m * W_m = b_m and then solve by finding the inverse of A_m i.e. W_m = Inv(A_m) * b_m
    num_assets = covariance_matrix.shape[0]

    if mean_returns is None:
        solve_minimum_variance = True
        num_constraints = 1
    else:
        solve_minimum_variance = False
        num_constraints = 2
        if return_target is None:
            return_target = max(mean_returns)

    # setup matrix A
    if not solve_minimum_variance:
        matrix_a = np.vstack([covariance_matrix, mean_returns, [1.] * num_assets])
        matrix_a = np.hstack([matrix_a, [[r_mean] for r_mean in mean_returns] + num_constraints * [[0.]]])
    else:
        matrix_a = np.vstack([covariance_matrix, [1.] * num_assets])
    matrix_a = np.hstack([matrix_a, [[1.]] * num_assets + num_constraints * [[0.]]])

    # setup vector b
    if solve_minimum_variance:
        vector_b = [[0.]] * num_assets + [[allocation]]
    else:
        vector_b = [[0.]] * num_assets + [[return_target], [allocation]]

    # solve A_m * z_v = b_v
    inv_matrix_a = np.linalg.inv(matrix_a)
    solution_vector_z = np.matmul(inv_matrix_a, vector_b)  # z_v = Inv(A_m) * b_v
    optimal_weights = solution_vector_z[:-num_constraints]  # ignore the Lagrangian multipliers
    return np.transpose(optimal_weights)[0]


def theoretical_mean_variance_portfolio_weights(mean_returns: np.array, covariance_matrix: np.array, return_target: float,
                                                allocation: float = 1.0)->np.array:
    """
    Finds the theoretical weights of the portfolio with the minimum variance that satisfy the return target.
    The only constraint that the sum of all weights should be equal to a given allocation.
    This is done by solving the linear equations derived from setting the gradient of the Lagrangian to the zero vector.
    :param mean_returns: array
    :param covariance_matrix: array
    :param return_target: float
    :param allocation: float
    :return: np.array
    """
    return _theoretical_mean_variance_optimizer(mean_returns, covariance_matrix, return_target, allocation)


def theoretical_minimum_variance_portfolio_weights(covariance_matrix: np.array, allocation: float = 1.0)->np.array:
    """
    Finds the theoretical weights of the minimum variance portfolio with the only constraint that the sum of all weights
    should be equal to a given allocation (this is the only constraint for the optimization problem).
    This is done by solving the linear equations derived from setting the gradient of the Lagrangian to the zero vector.
    :param covariance_matrix: array
    :param allocation: float
    :return: np.array
    """
    return _theoretical_mean_variance_optimizer(None, covariance_matrix, None, allocation)


def efficient_frontier(mean_returns: np.array, covariance_matrix: np.array, allocation: float = 1.0):
    """
    Returns the weight, return and volatility of portfolios across the efficient frontier
    :param mean_returns: np.array
    :param covariance_matrix: np.array
    :param allocation: float
    :return: (np.array, np.array, np.array)
    """

    if mean_returns.shape[0] < 2 or covariance_matrix.shape[0] < 2:
        raise ValueError('Need at least two assets.')
    if max(mean_returns) == min(mean_returns):
        raise ValueError('The mean returns cannot all be the same.')
    # calculate two efficient portfolios with the target returns set to the two provided mean returns
    min_var_weights = theoretical_minimum_variance_portfolio_weights(covariance_matrix, allocation)
    efficient_weights = theoretical_mean_variance_portfolio_weights(mean_returns, covariance_matrix, max(mean_returns), allocation)

    # calculate the efficient frontier by looking at linear combinations of an efficient portfolio and the minimum
    # variance portfolio
    alpha_list = np.linspace(1.0, -1.0)  # alpha * minimum variance + (1 - alpha) * efficient portfolio
    portfolio_weight = np.array([])
    portfolio_return = np.array([])
    portfolio_vol = np.array([])
    for alpha in alpha_list:
        new_weight = alpha * min_var_weights + (1 - alpha) * efficient_weights
        new_return = np.matmul(new_weight, np.transpose(mean_returns))
        new_vol = portfolio_volatility(new_weight, covariance_matrix)
        portfolio_weight = np.concatenate([portfolio_weight, new_weight])
        portfolio_return = np.concatenate([portfolio_return, [new_return]])
        portfolio_vol = np.concatenate([portfolio_vol, [new_vol]])
    return portfolio_weight, portfolio_return, portfolio_vol


def rolling_mean_variance_implementation(strategy_returns: pd.DataFrame, rolling_window: int, min_total_weight: float = 0.,  max_total_weight: float = 1.,
                                         min_instrument_weight: {float, list, tuple}=0., max_instrument_weight: {float, list, tuple}=1.,
                                         risk_aversion_factor: int = 1, return_target: float = 0):
    # check inputs
    if rolling_window < 2 or rolling_window > strategy_returns.shape[0]:
        raise ValueError('rolling_window needs to be larger or equal to 2 and smaller or equal to the length of the '
                         'strategy_return DataFrame ({}).'.format(strategy_returns.shape[0]))
    strategy_returns = strategy_returns.copy()
    calendar = strategy_returns.index
    strategy_returns.reset_index(inplace=True, drop=True)
    weight_result = None  # initialize the result
    for row in strategy_returns.itertuples():
        index_i = row.Index
        print(progression_bar_str(index_i + 1, len(calendar)))
        if index_i >= rolling_window:
            # select a subsection of the DataFrame and calculate the mean and covariance
            strategy_returns_sub = strategy_returns.iloc[index_i - rolling_window:index_i, :]
            mean_returns = strategy_returns_sub.mean().values
            covariance = np.cov(np.transpose(strategy_returns_sub.values).astype(float))

            # perform mean variance optimization
            optimal_weights = optimal_mean_variance_portfolio_weights_with_constraints(mean_returns, covariance, weight_result[index_i - 1, :], min_total_weight, max_total_weight, min_instrument_weight, max_instrument_weight, risk_aversion_factor, return_target)

            weight_result = np.vstack([weight_result, optimal_weights])
        else:
            # in the beginning, use equal weights
            equal_weight = np.array(strategy_returns.shape[1] * [1 / strategy_returns.shape[1]])
            if weight_result is None:
                weight_result = equal_weight
            else:
                weight_result = np.vstack([weight_result, equal_weight])  # add a row to the array
    return pd.DataFrame(data=weight_result, index=calendar, columns=list(strategy_returns))


def main():
    cov_matrix = np.array([[0.01, 0.0018, 0.0011], [0.0018, 0.0109, 0.0026], [0.0011, 0.0026, 0.0199]])
    mean_returns = np.array([0.0427, 0.0015, 0.0285])
    volatility = np.sqrt(np.array([cov_matrix[0, 0], cov_matrix[1, 1], cov_matrix[2, 2]]))

    print('Weights for the minimum variance portfolio')
    print('Analytical solution')
    print(theoretical_minimum_variance_portfolio_weights(cov_matrix))

    print('Numerical solution')
    minimum_variance_portfolio_weights_with_constraints(cov_matrix, max_instrument_weight=9999.0, min_instrument_weight=-9999.0)

    # # calculate the efficient frontier
    # efficient_frontier_result = efficient_frontier(mean_returns, cov_matrix)
    # efficient_returns = efficient_frontier_result[1]
    # efficient_vols = efficient_frontier_result[2]
    # total_returns = np.concatenate([mean_returns, efficient_returns])
    # total_vols = np.concatenate([volatility, efficient_vols])
    #
    # # plot results
    # plt.scatter(total_vols, total_returns)
    # plt.xlim(0, 0.15)
    # plt.ylim(0, 0.06)
    # plt.show()


if __name__ == '__main__':
    main()

