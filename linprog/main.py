from simplex_method import simplex_method
import numpy as np
from intvalpy  import lineqs
import matplotlib as plt
from gilp.simplex import LP
import gilp

# c = [-8,-10, -7]
# A = [[1, 3, 2], [1, 5, 3]]
# b = [10, 8]

c = np.array([5, 3])
A = np.array([[2, 1],
              [1, 1],
              [1, 0]])
b = np.array([20, 16, 7])

print(simplex_method(c, A, b))

b = np.transpose(b)
c = np.transpose(c)

lp = LP(A,b,c)
# print(gilp.visualize.feasible_region(lp))
gilp.visualize.simplex_visual(lp).show()


# m = size
# print(lineqs(-A, -b))
