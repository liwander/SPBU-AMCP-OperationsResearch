import pulp
import highspy
from mip import Model

# solver=pulp.getSolver()
h = Model(solver_name='CBC')
h.read(filename)
print(h)