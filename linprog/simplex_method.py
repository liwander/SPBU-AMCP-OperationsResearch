import numpy as np

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


    print(tableau)
    print("=" * 10)
    while np.any(tableau[0] > 0):
        pivot_col = np.argmax(tableau[0, 0:])
        pivot_row = np.argmin(tableau[1:, -1] / tableau[1:, pivot_col]) + 1
        print(pivot_row, pivot_col)

        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element

        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
                print(tableau)
                print('=' * 10)

    optimal_solution = tableau[0, -1]
    solution = np.array([])
    for j in range(n):
        xpos = np.where(tableau[1:, j] == 1)[0]
        if xpos.size == 0:
            solution = np.append(solution, 0)
        else:
            solution = np.append(solution, tableau[int(xpos[0]), -1])

    return optimal_solution, solution

