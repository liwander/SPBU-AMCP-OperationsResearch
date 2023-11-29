from simplex_method import simplex_method


c = [-8,-10, -7]
A = [[1, 3, 2], [1, 5, 3]]
b = [10, 8]
print(simplex_method(c, A, b))