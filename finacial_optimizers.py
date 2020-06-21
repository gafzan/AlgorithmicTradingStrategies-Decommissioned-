"""
financial_optimizers.py
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


def total_allocation_constraint(weight, allocation: float):
    return weight.sum() - allocation


def portfolio_variance(weights: np.array, covariance_matrix: np.array):
    """

    :param weights:
    :param covariance_matrix:
    :return:
    """
    return np.matmul(np.matmul(weights, covariance_matrix), np.transpose(weights))


def portfolio_volatility(weights: np.array, covariance_matrix: np.array):
    """

    :param weights:
    :param covariance_matrix:
    :return:
    """
    port_variance = portfolio_variance(weights, covariance_matrix)
    return np.sqrt(port_variance)


def minimum_variance_portfolio_weights_with_constraints(covariance_matrix, initial_guess: np.array = None,
                                                        min_total_weight: float = 1., max_total_weight: float = 1.,
                                                        min_instrument_weight: float = 0., max_instrument_weight: float = 1.):

    num_assets = covariance_matrix.shape[0]
    if initial_guess is None:  # if not initial guess is provided, use equal weighting as an initial guess
        initial_guess = np.array(num_assets * [1 / num_assets])

    instrument_weight_bounds = tuple(num_assets * [(min_instrument_weight, max_instrument_weight)])
    max_total_weight_cons = {'type': 'eq',  # TODO change to ineq
                             'fun': total_allocation_constraint,
                             'args': (max_total_weight, )}
    min_total_weight_cons = {'type': 'ineq',  # TODO change to ineq
                             'fun': total_allocation_constraint,
                             'args': (min_total_weight,)}
    cons = [max_total_weight_cons, min_total_weight_cons]
    sol = minimize(portfolio_variance, initial_guess, method='SLSQP', args=(covariance_matrix, ),
                   bounds=instrument_weight_bounds, constraints=cons)
    print(sol.x)


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

