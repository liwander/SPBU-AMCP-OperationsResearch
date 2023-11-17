import jade
import numpy as np

def goldstein_price(varg : np.ndarray):
    x = varg[0]
    y = varg[1]
    term1 = 1 + ((x + y + 1)**2) * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
    term2 = 30 + ((2*x - 3*y)**2) * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
    return term1 * term2

params = jade.get_default_params(2)


population_size = params['population_size']
individual_size = params['individual_size']
bounds = np.array([[-2, 2], [-2, 2]])
func = goldstein_price
opts = None
p = params['p']
c = params['c']
callback = params['callback'] 
max_evals = params['max_evals']
seed = None

res = jade.apply(population_size, individual_size, bounds, func, opts, p, c, callback, max_evals, seed)
print(f'f minimum: {res[1]} at {res[0]}')