import matplotlib.pyplot as plt
import numpy as np

def plot_alg_analytics(jade_alg_analytics, filename='jade_analytics_data.png'):
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(len(jade_alg_analytics), 1, hspace=0.5)
    axs = gs.subplots()

    for plot_ix, jade_alg_analytics in enumerate(jade_alg_analytics.items()):
        analytics_name = jade_alg_analytics[0]
        data = jade_alg_analytics[1]
        axs[plot_ix].set_title(analytics_name)
        axs[plot_ix].scatter(data[:, 0], data[:, 1], c=np.linspace(0, 1, data.shape[0]))
        axs[plot_ix].set_xlim(-2, 2)
        axs[plot_ix].set_ylim(-2, 2)
        axs[plot_ix].set_xlabel('x')
        axs[plot_ix].set_ylabel('y')
        arrowprops = dict(arrowstyle='->', connectionstyle='arc3', facecolor='black')
        axs[plot_ix].annotate('first iter', (data[0, 0], data[0, 1]), xytext=(-1.70, 1.70), arrowprops=arrowprops)
        axs[plot_ix].annotate('last iter', (data[-1, 0], data[-1, 1]), xytext=(1.70, -1.70), arrowprops=arrowprops)

    plt.savefig(filename)


