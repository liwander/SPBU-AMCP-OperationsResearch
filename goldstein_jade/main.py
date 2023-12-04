import jade
import matplotlib.pyplot as plt
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


alg_analytics = {'median individ' :res[2], 
                 'mean individ':res[3]}

fig = plt.figure()
gs = fig.add_gridspec(len(alg_analytics), 1, hspace=0.5)
axs = gs.subplots()

for plot_ix, alg_analytics in enumerate(alg_analytics.items()):
    analytics_name = alg_analytics[0]
    data = alg_analytics[1]
    axs[plot_ix].set_title(analytics_name)
    axs[plot_ix].scatter(data[:, 0], data[:, 1], c=np.linspace(0, 1, data.shape[0]))
    axs[plot_ix].set_xlim(-2, 2)
    axs[plot_ix].set_ylim(-2, 2)
    axs[plot_ix].set_xlabel('x')
    axs[plot_ix].set_ylabel('y')
    arrowprops = dict(arrowstyle='->', connectionstyle='arc3', facecolor='black')
    axs[plot_ix].annotate('first iter', (data[0, 0], data[0, 1]), xytext=(-1.70, 1.70), arrowprops=arrowprops)
    axs[plot_ix].annotate('last iter', (data[-1, 0], data[-1, 1]), xytext=(1.70, -1.70), arrowprops=arrowprops)

plt.savefig('rand_run.png')

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

fig = plt.figure()
gs = fig.add_gridspec(len(marathon_avg_run), 1, hspace=0.5)
axs = gs.subplots()

for plot_ix, alg_analytics in enumerate(marathon_avg_run.items()):
    analytics_name = alg_analytics[0]
    data = alg_analytics[1]
    axs[plot_ix].set_title(analytics_name)
    axs[plot_ix].scatter(data[:, 0], data[:, 1], c=np.linspace(0, 1, data.shape[0]))
    axs[plot_ix].set_xlim(-2, 2)
    axs[plot_ix].set_ylim(-2, 2)
    axs[plot_ix].set_xlabel('x')
    axs[plot_ix].set_ylabel('y')
    arrowprops = dict(arrowstyle='->', connectionstyle='arc3', facecolor='black')
    axs[plot_ix].annotate('first iter', (data[0, 0], data[0, 1]), xytext=(-1.70, 1.70), arrowprops=arrowprops)
    axs[plot_ix].annotate('last iter', (data[-1, 0], data[-1, 1]), xytext=(1.70, -1.70), arrowprops=arrowprops)

plt.savefig('marathon_avg_run.png')