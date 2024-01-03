import argparse
import os
import random
from collections import OrderedDict

import botorch
import numpy as np
import torch

from .function_realworld_bo.functions_mujoco import *
from .function_realworld_bo.functions_realworld_bo import *
from .functions_bo import *
from .highdim_functions import *
from .lasso_benchmark import *


def get_arguments():
    parser = argparse.ArgumentParser(description='Process inputs')
    parser.add_argument('-f', '--func', help='specify the test function')
    parser.add_argument('-d', '--dim', type=int, help='specify the problem dimensions', default=10)
    parser.add_argument('-n', '--maxevals', type=int, help='specify the maxium number of evaluations to collect in the search')
    parser.add_argument('--solver', type=str, help='specify the solver', default='bo')
    parser.add_argument('--seed', type=int, help='seeding option', default=1)


    args = parser.parse_args()


    dim = args.dim
    func = args.func.lower()
    if func == 'ackley':
        f = ackley(dim)
    elif func == 'shifted-ackley':
        f = shifted_ackley(dim)
    elif func == 'rastrigin':
        f = rastrigin(dim)
    elif func == 'ellipsoid':
        f = ellipsoid(dim)
    elif func == 'shifted-ellipsoid':
        f = shifted_ellipsoid(dim)
    elif func == 'levy':
        f = Levy(dim)
    elif func == 'shifted-levy':
        f = shifted_levy(dim)
    elif func == 'schwefel':
        f = schwefel(dim)
    elif func == 'alpine':
        f = alpine(dim)
    elif func == 'shifted-alpine':
        f = shifted_alpine(dim)
    elif func == 'eggholder':
        f = egg_holder()
        dim = f.input_dim
    elif func == 'beale':
        f = beale()
        dim = f.input_dim
    elif func == 'branin':
        f = branin_uniformbound()
        dim = f.input_dim
    elif func == 'rosenbrock':
        f = rosenbrock(dim)
    elif func == 'powell':
        f = powell(dim)
    elif func == 'schaffer':
        f = schaffer_n2()
    elif func == 'robot-pushing':
        f = Robot_pushing()
    elif func == 'rover60':
        f = Rover()
    elif func == 'rover20':
        f = Rover20()
    elif func == 'rover100':
        f = Rover100()
    elif func == 'lunar-landing':
        f = Lunar_landing()
    elif func == 'bipedal-walking':
        f = Bipedal_walking()
    elif func == 'electron9':
        f = ElectronSphere9np()
    elif func == 'electron6':
        f = ElectronSphere6np()
    elif func == 'lasso-simple':
        f = LassoSimpleBenchmark()
    elif func == 'lasso-medium':
        f = LassoMediumBenchmark()
    elif func == 'lasso-high':
        f = LassoHighBenchmark()
    elif func == 'lasso-hard':
        f = LassoHardBenchmark()
    elif func == 'lasso-diabete':
        f = LassoDiabetesBenchmark()
    elif func == 'lasso-dna':
        f = LassoDNABenchmark()
    elif func == 'hartmann500':
        f = Hartmann500D()
    elif func == 'branin20':
        f = Branin20D()
    elif func == 'branin40':
        f = Branin40D()
    elif func == 'branin500':
        f = Branin500D()
    elif func == 'schaffer40':
        f = Schaffer40()
    elif func == 'schaffer100':
        f = Schaffer100()
    elif func == 'bohachevsky100':
        f = Bohachevsky100()
    elif func == 'mopta08':
        f = MoptaSoftConstraints()
    elif func == 'hopper':
        f = Hopper()
    elif func == 'walker2d':
        f = Walker2d()
    elif func == 'half-cheetah':
        f = HalfCheetah()
    elif func == 'humanoid':
        f = Humanoid()
    elif func == 'ant':
        f = Ant()
    elif func == 'swimmer':
        f = Swimmer()
    elif func == 'svm':
        f = SVMBenchmark()
    else:
        raise NotImplementedError(f'Objective function {func} is not supported')
    
    dim = f.input_dim
    max_evals = args.maxevals
    dict = {
        'func_name': func,
        "f": f,
        'max_evals': max_evals,
        'solver': args.solver,
        'seed': args.seed,
    }
    return dict

def get_bound(bounds):
    if isinstance(bounds, OrderedDict):
        return np.array([val for val in bounds.values()])
    else:
        return np.array(bounds)
    
def set_seed(seed=1234):
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    botorch.manual_seed(seed)