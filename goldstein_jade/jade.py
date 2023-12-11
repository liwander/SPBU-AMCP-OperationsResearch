import commons
import numpy as np
from typing import Callable, Union, Dict, Any


def get_default_params(dim: int) -> dict:
    """
        Returns the default parameters of the JADE Differential Evolution Algorithm.
        :param dim: Size of the problem (or individual).
        :type dim: int
        :return: Dict with the default parameters of the JADE Differential
        Evolution Algorithm.
        :rtype dict
        """
    pop_size = 20*dim
    return {'max_evals': 1_0000*dim, 'individual_size': dim, 'callback': None,
            'population_size': pop_size, 'c': 0.1, 'p': max(.05, 3/pop_size), 'seed': None}


def apply(population_size: int, individual_size: int, bounds: np.ndarray,
          func: Callable[[np.ndarray], float], opts: Any,
          p: Union[int, float], c: Union[int, float], callback: Callable[[Dict], Any],
          max_evals: int, seed: Union[int, None]) -> [np.ndarray, int]:
    """
    Applies the JADE Differential Evolution algorithm.
    :param population_size: Size of the population.
    :type population_size: int
    :param individual_size: Number of gens/features of an individual.
    :type individual_size: int
    :param bounds: Numpy ndarray with individual_size rows and 2 columns.
    First column represents the minimum value for the row feature.
    Second column represent the maximum value for the row feature.
    :type bounds: np.ndarray
    :param func: Evaluation function. The function used must receive one
     parameter.This parameter will be a numpy array representing an individual.
    :type func: Callable[[np.ndarray], float]
    :param opts: Optional parameters for the fitness function.
    :type opts: Any type.
    :param p: Parameter to choose the best vectors. Must be in (0, 1].
    :type p: Union[int, float]
    :param c: Variable to control parameter adoption. Must be in [0, 1].
    :type c: Union[int, float]
    :param callback: Optional function that allows read access to the state of all variables once each generation.
    :type callback: Callable[[Dict], Any]
    :param max_evals: Number of evaluations after the algorithm is stopped.
    :type max_evals: int
    :param seed: Random number generation seed. Fix a number to reproduce the
    same results in later experiments.
    :type seed: Union[int, None]
    :return: A pair with the best solution found and its fitness.
    :rtype [np.ndarray, int]
    """
    # 0. Check parameters are valid
    if type(population_size) is not int or population_size <= 0:
        raise ValueError("population_size must be a positive integer.")

    if type(individual_size) is not int or individual_size <= 0:
        raise ValueError("individual_size must be a positive integer.")

    if type(max_evals) is not int or max_evals <= 0:
        raise ValueError("max_evals must be a positive integer.")

    if type(bounds) is not np.ndarray or bounds.shape != (individual_size, 2):
        raise ValueError("bounds must be a NumPy ndarray.\n"
                         "The array must be of individual_size length. "
                         "Each row must have 2 elements.")

    if type(seed) is not int and seed is not None:
        raise ValueError("seed must be an integer or None.")

    if type(p) not in [int, float] and 0 < p <= 1:
        raise ValueError("p must be a real number in (0, 1].")
    if type(c) not in [int, float] and 0 <= c <= 1:
        raise ValueError("c must be an real number in [0, 1].")

    np.random.seed(seed)

    # 1. Init population
    population = commons.init_population(population_size, individual_size, bounds)
    u_cr = 0.5
    u_f = 0.6

    p = np.ones(population_size) * p
    fitness = commons.apply_fitness(population, func, opts)
    max_iters = max_evals // population_size
    # max_iters = 10_000
    avg, median = [],[]
    fbest=[]
    prev_moving_avg=np.zeros(shape=(1, individual_size))
    cur_moving_avg=prev_moving_avg
    for current_generation in range(max_iters):

        # 2.1 Generate parameter values for current generation
        cr = np.random.normal(u_cr, 0.1, population_size)
        f = np.random.rand(population_size // 3) * 1.2
        f = np.concatenate((f, np.random.normal(u_f, 0.1, population_size - (population_size // 3))))

        # 2.2 Common steps
        mutated = commons.current_to_pbest_mutation(population, fitness, f.reshape(len(f), 1), p, bounds)
        crossed = commons.crossover(population, mutated, cr.reshape(len(f), 1))
        c_fitness = commons.apply_fitness(crossed, func, opts)
        population, indexes = commons.selection(population, crossed,
                                                      fitness, c_fitness, return_indexes=True)

        # 2.3 Adapt for next generation
        if len(indexes) != 0:
            u_cr = (1 - c) * u_cr + c * np.mean(cr[indexes])
            u_f = (1 - c) * u_f + c * (np.sum(f[indexes]**2) / np.sum(f[indexes]))

        fitness[indexes] = c_fitness[indexes]
        if callback is not None:
            callback(**(locals()))

        median_func = np.median(fitness, axis=0)
        median.append(median_func)
        avg_func = np.mean(fitness, axis=0)
        avg.append(avg_func)
        # print(median_individ.shape)
        fbest.append(fitness[np.argmin(fitness)])
        print(fitness[np.argmin(fitness)])

        cur_moving_avg=np.ma.average(np.array(population),axis=0)
        if np.linalg.norm(cur_moving_avg-prev_moving_avg)<1e-6:
            break
        else:
            prev_moving_avg = cur_moving_avg
            
    print('\n\n\n ', current_generation)
    best = np.argmin(fitness)
    return population[best], fitness[best], np.array(median), np.array(fbest)
