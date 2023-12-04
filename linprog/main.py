from simplex_method import simplex_method
from visiualization import simplex_method_visualization
import numpy as np
from gilp.simplex import LP
import gilp

# c = [-8,-10, -7]
# A = [[1, 3, 2], [1, 5, 3]]
# b = [10, 8]

c = np.array([2, 3, 4])
A = np.array([[3, 2, 1],
              [2, 5, 3]])
b = np.array([10, 15])

# print(simplex_method(c, A, b))

dim = c.shape[0]
A_all = -np.vstack((A, -np.eye(dim)))
B_all = -np.hstack((b, np.zeros(dim)))

# print(A_all)

b = np.transpose(b)
c = np.transpose(c)
lp = LP(A,b,c)
# print(gilp.visualize.feasible_region(lp))
# gilp.visualize.simplex_visual(lp).show()

simplex_method_visualization(A_all, B_all, simplex_method(c, A, b))