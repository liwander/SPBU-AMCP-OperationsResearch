import numpy as np
from mip import Model
from scipy.sparse import csr_matrix

def read_mps(file: str):
    """
    Reads a .mps and saves all the data of the MILP:

    min c^T * x

    s.t. b_l <= A*x <= b_u
          lb <=   x <= ub
                x_i integer if integrality[i] = 1
    """
    mdl = Model(solver_name="CBC")
    mdl.verbose=0
    mdl.read(file)

    # model parameters
    num_vars = len(mdl.vars)
    num_cons = len(mdl.constrs)

    # variable types and bounds
    lb = np.zeros(num_vars)
    ub = np.inf*np.ones(num_vars)
    integrality = np.zeros(num_vars)
    for i, var in enumerate(mdl.vars):
        lb[i] = var.lb
        ub[i] = var.ub
        if var.var_type != "C":
            integrality[i] = 1

    # objective
    c = np.zeros(num_vars)
    for i, var in enumerate(mdl.vars):
        if var in mdl.objective.expr:
            c[i] = mdl.objective.expr[var]
    # print(mdl.sense)
    if mdl.sense != "MAX":
        c *= -1.0

    # constraint coefficient matrix
    b_l = -np.inf*np.ones((num_cons))
    b_u = np.inf*np.ones((num_cons))
    row_ind = []
    col_ind = []
    data    = []
    for i, con in enumerate(mdl.constrs):
        if con.expr.sense == "=":
            b_l[i] = con.rhs
            b_u[i] = con.rhs
        elif con.expr.sense == "<":
            b_u[i] = con.rhs
        elif con.expr.sense == ">":
            b_l[i] = con.rhs
        for j, var in enumerate(mdl.vars):
            if var in (expr := con.expr.expr):
                coeff = expr[var]
                row_ind.append(i)
                col_ind.append(j)
                data.append(coeff)
    A = csr_matrix((data, (row_ind, col_ind)), shape=(num_cons, num_vars))
    return c, b_l, A, b_u, lb, ub, integrality

