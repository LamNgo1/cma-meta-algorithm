# coding: utf-8
import time

import numpy as np

from test_functions.utils import get_arguments, get_bound, set_seed
from cmabo.cma_bo import CMABayesianOptimization

# parameters
input_dict = get_arguments()
f = input_dict['f']
MAX_EVALS = input_dict['max_evals']
solver = input_dict['solver']
seed = input_dict['seed']

print(f'CMA-{solver.upper()}: {f.name}-{f.input_dim}D function with max_evals={MAX_EVALS} and seed={seed}')

# Start
bounds = get_bound(f.bounds)
lb = bounds[:, 0]
ub = bounds[:, 1]

history_fx = np.zeros((MAX_EVALS, 0))
set_seed(seed=seed)
print(f'==============> seed={seed} <===============')
stamp1 = time.time()
cmabo = CMABayesianOptimization(n_init=20, f=f.func, solver=solver, lb=lb, ub=ub,
                                max_evals=MAX_EVALS, func_name=f.name, keep_record=True 
                                )
cmabo.optimize()
stamp2 = time.time()
cmabo._dumpdata(seed, total_time=stamp2-stamp1)
experiment = f'{f.name}_{f.input_dim}d'
history_fx = np.hstack((history_fx, cmabo.observed_fx[:MAX_EVALS]))

np.savetxt(f'cma-{solver}_{experiment}_{seed}.csv', history_fx, delimiter=',')

print('------> FINISHED <---------')
print(f'CMA-{solver.upper()}: {f.name}-{f.input_dim}D function with max_evals={MAX_EVALS} and seed={seed}')
