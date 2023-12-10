from simplex_method import simplex_method
from visiualization import simplex_method_visualization
import numpy as np
from gilp.simplex import LP
import gilp
from read_problem_coefs import read_mps

c = np.array([2,3, 4])
A = np.array([[3, 2, 1], [2, 5, 3]])
b = np.array([10, 15])

# c = np.array([2, 3, 4])
# A = np.array([[3, 2,  1],
#               [2,5, 3]])
# b = np.array([10, 15])


# c = np.array([-1, -1])
# A = np.array([[-1, 0], [0, -1], [1, 1]])
# b = np.array([-6, -6, 11])

# print()
simplex_method_visualization(A, b, simplex_method(c, A, b))
# print(gilp.visualize.feasible_region(lp))

dim = c.shape[0]
A_all = -np.vstack((A, -np.eye(dim)))
B_all = -np.hstack((b, np.zeros(dim)))

# # print(A_all)

# b = np.transpose(b)
# c = np.transpose(c)
# lp = LP(A,b,c)
# gilp.visualize.simplex_visual(lp).show()

simplex_method_visualization(A_all, B_all, simplex_method(c, A, b))

filedir = '/home/anver/Projects/OperationsResearch/SPBU-AMCP-OperationsResearch/linprog/test_problems/'
filenames = ['b-ball.mps', 'enlight8.mps', 'neos-1425699.mps']

for filename in filenames:

    c, bl, A, bu, lb, lu, integrality = read_mps(filedir + filename)

    print("file was read")
    A = A.A
    A = np.vstack((A, -A))
    b = np.hstack((bu, bl))

    # print(c)
    # print(A)
    # print(b)

    print(simplex_method(c, A, b)[-1])