import jade
import numpy as np
from visualization import plot_alg_analytics
from functools import reduce
from operator import mul

def goldstein_price(varg : np.ndarray):
    x = varg[0]
    y = varg[1]
    term1 = 1 + ((x + y + 1)**2) * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
    term2 = 30 + ((2*x - 3*y)**2) * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
    return term1 * term2

def myf(x):
    return reduce(mul, x)

def rosenbrock(args):
    sum = 0
    for i in range(len(args)-1):
        sum += 100*(args[i+1] - args[i]**2)**2 + (1 - args[i])**2
    return sum


def sphere(args):
    return sum(map(lambda x: x**2, args))

dim = int(1000)
params = jade.get_default_params(dim)


population_size = params['population_size']
individual_size = params['individual_size']
bounds = np.array([[-1e2, 1e2]] * dim)
# print(bounds)
func = sphere
opts = None
p = params['p']
c = params['c']
callback = params['callback'] 
max_evals = params['max_evals']
seed = None

res = jade.apply(population_size, individual_size, bounds, func, opts, p, c, callback, max_evals, seed)
print(f'f minimum: {res[1]} at {res[0]}')

## random run data
rand_run_alg_analytics = {'median individ' :res[2], 
                 'mean individ':res[3]}

# print(rand_run_alg_analytics['mean individ'])

# plot_alg_analytics(rand_run_alg_analytics, filename='random_run.png')


## several runs average data
marathon_len = 10
marathon_analytics = {'median' : np.ndarray((marathon_len, max_evals // population_size, individual_size)),
                      'mean' : np.ndarray((marathon_len, max_evals // population_size, individual_size))}

for run_number in range(marathon_len):
    run_res = jade.apply(population_size, individual_size, bounds, func, opts, p, c, callback, max_evals, seed)
    marathon_analytics['median'][run_number] = run_res[2]
    marathon_analytics['mean'][run_number] = run_res[3]

marathon_median = np.mean(marathon_analytics['median'][:], axis=0)
marathon_mean = np.mean(marathon_analytics['mean'][:], axis=0) 

marathon_avg_run = {'mean_individ': marathon_mean,
                     'median_individ': marathon_median}


# plot_alg_analytics(marathon_avg_run, filename='marathon_avg_run')