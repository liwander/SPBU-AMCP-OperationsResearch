import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
from numpy.linalg import solve

def current_optimal_solution(tableau):
    optimal_solution = tableau[0, -1]
    opt_arg = np.array([])
    for j in range(tableau.shape[1] - tableau.shape[0]):
        xpos = np.where(tableau[1:, j] == 1)[0]
        if xpos.size == 0:
            opt_arg = np.append(opt_arg, 0)
        else:
            opt_arg = np.append(opt_arg, tableau[int(xpos[0]) + 1, -1])
    return optimal_solution, opt_arg
        
def simplex_method(c, A, b):
    c = np.array(c)
    A = np.array(A)
    b = np.array(b)

    m, n = A.shape
    # Add slack variables to convert inequalities to equalities
    A_slack = np.hstack((A, np.eye(m)))
    c_slack = np.hstack((c, np.zeros(m)))

    # Initialize the tableau
    tableau = np.vstack((np.hstack((-c_slack, 0)), np.hstack((A_slack, b[:, None]))))


    sepline = '=' * 10 + '\n'
    print(tableau)
    # print(sepline)
    maxiter = 10
    it = 0
    res = []
    res = current_optimal_solution(tableau)
    print(res)

    # print(tableau[0, :-(1 + len(b))])
    while np.any(tableau[0, 0:-1] < 0):
        it += 1
        pivot_col = np.argmin(tableau[0, 0:-1])
        indicator_col = tableau[1:, -1] / tableau[1:, pivot_col]
        pivot_row = np.where(indicator_col > 0, indicator_col, np.inf).argmin() + 1
        # print(f"iteration initial tableau: \n{tableau}")
        # print(f'indicator column: \n{indicator_col}')

        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element
        # print(f"tableau after indenting pivot row: \n{tableau}")


        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
                # print(f"tableau after pivoting:  \n{tableau}")

        # print()
        # print(tableau)
        res = current_optimal_solution(tableau)
        # print(res)

    return res
