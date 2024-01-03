# coding: utf-8
import time

from cmabo.cma_bo import CMABayesianOptimization
from test_functions.utils import get_arguments, get_bound, set_seed

# parameters
input_dict = get_arguments()
objective = input_dict['f']
max_evals = input_dict['max_evals']
solver = input_dict['solver']
seed = input_dict['seed']

information = f'CMA-{solver.upper()}: {objective.name}-{objective.input_dim}D function with max_evals={max_evals} and seed={seed}'
print(information)
print(f'==============> seed={seed} <===============')
set_seed(seed=seed)
# Start
bounds = get_bound(objective.bounds)
lb = bounds[:, 0]
ub = bounds[:, 1]

stamp1 = time.time()
cmabo = CMABayesianOptimization(n_init=20, f=objective.func, solver=solver, lb=lb, ub=ub,
                                max_evals=max_evals, func_name=objective.name, keep_record=True 
                                )
cmabo.optimize()
stamp2 = time.time()
cmabo.dumpdata(seed, total_time=stamp2-stamp1) # Save pickle file

print('------> FINISHED <---------')
print(information)
