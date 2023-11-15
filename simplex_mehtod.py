import numpy as np

def simplex_method(c, A, b):
    A = np.array(A)
    c = np.array(c)
    b = np.array(b)
    m, n = A.shape

    # Add slack variables to convert inequalities to equalities
    A = np.hstack((A, np.eye(m)))
    c = np.hstack((c, np.zeros(m)))

    # Initialize tableau
    tableau = np.vstack((np.hstack((np.array([0]), -c, 1)),
                         np.hstack((b[:, None], A, np.zeros(m)[:, None]))))
    print(tableau)

    while np.any(tableau[0, 1:] < 0):
        # Find entering variable (most negative coefficient in objective function)
        entering_var = np.argmin(tableau[0, 1:])

        # Find leaving variable (minimum ratio test)
        ratios = tableau[1:, 0] / tableau[1:, entering_var+1]
        leaving_var = np.argmin(ratios)

        # Pivot operation
        pivot_row = tableau[leaving_var+1, :]
        tableau = tableau - np.outer(tableau[:, entering_var+1], pivot_row) / pivot_row[entering_var+1]
        tableau[leaving_var+1, :] = pivot_row / pivot_row[entering_var+1]

    return tableau[0, 0], tableau[1:, 0]


#You can use this implementation by passing the objective function coefficients c,
#  the constraint matrix A, and the right-hand side vector b as input.
#  The function will return the optimal value of the objective function
#  and the optimal solution vector.
