'''Code implementation for CMA Meta-Algorithm'''
import datetime
import logging
import math
import os
import pickle
import time
import warnings
from copy import deepcopy
from logging import debug
from typing import List, Union

import cma
import gpytorch
import numpy as np
import scipy
import torch
from botorch.exceptions import BotorchWarning

from baxus_cma.baxus import BAxUS
from baxus_cma.benchmarks.benchmark_function import Benchmark
from baxus_cma.util.behaviors import BaxusBehavior
from baxus_cma.util.utils import to_1_around_origin
from turbo_cma.turbo_1 import Turbo1

from .gp import train_gp
from .utils import from_unit_cube, latin_hypercube, to_unit_cube

warnings.simplefilter("ignore", BotorchWarning)

class CMABayesianOptimization:
    '''The CMA-Meta-Algorithm for Bayesian Optimization

    Parameters
    ----------
    f: objective function handle
    lb: lower variable bounds, numpy.array, shape (d,)
    ub: upper variable bounds, numpy.array, shape (d,)
    max_evals: maximum evaludation budget, int
    solver: BO optimizers to be incorporated, string, choose one of {'bo', 'turbo', 'baxus'}
    n_init: number of initial points, int
    func_name: function name (for logging output only)
    keep_record: whether to store pkl files 
    ----------
    '''
    def __init__(
        self, 
        f,                      # objective function
        lb,                     # lower bound
        ub,                     # upper bound
        max_evals,              # maximum evaluation budget
        solver='baxus',         # 'bo', 'turbo' or 'baxus'
        n_init=20,
        func_name='',           # function name (for logging output only)
        keep_record=False,      # whether to store pkl files 
    ):
        assert len(lb) == len(ub)
        self.dim = len(lb)
        self.history_x = np.zeros((0, self.dim))   # dataset from previous restarts
        self.history_fx = np.zeros((0, 1))    # dataset from previous restarts
        self.observed_x = np.zeros((0, self.dim))  # working dataset, reset after restart
        self.observed_fx = np.zeros((0, 1))   # working dataset, reset after restart
        self.total_eval = 0                   # evaluation counter
        self.lb = lb                          # lower bound           
        self.ub = ub                          # upper bound
        self.f = f
        self.n_init = n_init
        self.max_evals = max_evals
        self.bound_tf = cma.BoundTransform([lb, ub])    # boundary handler for cma candidates
        self.lamda = 4 + math.floor(3*np.log(self.dim))      # population size
        self.es: cma.CMAEvolutionStrategy = None        # cma class
        self.max_cholesky_size = 2000
        self.std = np.sqrt(scipy.stats.chi2.ppf(q=0.9973,df=self.dim)) # 3-sigma rule
        self.solver = solver
        self.keep_record = keep_record
        self.func_name = func_name

        # turbo properties
        self.turbo_prev_length = 0.8
        self.turbo_succcount = 0
        self.turbo_failcount = 0
        self.turbo_prev_lb = lb
        self.turbo_prev_ub = ub
        self.turbo_restarted = False
        self.turbo_gp_init_x = np.zeros((0, self.dim)) 
        self.turbo_gp_init_fx = np.zeros((0, 1)) 

        # baxus properties
        self.baxus_prev_length = 0.8
        self.baxus_succcount = 0
        self.baxus_failcount = 0
        self.baxus_prev_lb = lb
        self.baxus_prev_ub = ub
        self.baxus_restarted = False
        self.baxus_restart_counter = 0
        self.baxus_prev_target_dim = 1
        self.baxus_prev_lengthscales = np.array([1.0])
        self.baxus_embedded_x = []
        self.baxus_embedded_fx = np.zeros((0, 1))
        self.baxus_projector = None
        self.baxus_trust_region_restarts = []
        self.baxus_dim_in_iterations = {}
        self.baxus_axus_change_iterations = []
        self.baxus_split_points = []


    def reset_info(self):
        # reset turbo info
        self.turbo_prev_length = 0.8
        self.turbo_succcount = 0
        self.turbo_failcount = 0
        self.turbo_prev_lb = self.lb
        self.turbo_prev_ub = self.ub
        self.turbo_restarted = False
        
        # reset baxus info
        self.baxus_prev_length = 0.8
        self.baxus_succcount = 0
        self.baxus_failcount = 0
        self.baxus_prev_lb = self.lb
        self.baxus_prev_ub = self.ub
        self.baxus_restarted = False
        self.baxus_prev_target_dim = 1
        self.baxus_prev_lengthscales = np.array([1.0])
        self.baxus_embedded_x = []
        self.baxus_embedded_fx = np.zeros((0, 1))
        self.baxus_projector = None
        self.baxus_trust_region_restarts = []
        self.baxus_dim_in_iterations = {}
        self.baxus_axus_change_iterations = []
        self.baxus_split_points = []

    def optimize(self):
        '''Run optimization'''
        self._initialize_storage()        
        uniformBound = all([self.lb[0]==_lb for _lb in self.lb]) and all([self.ub[0]==_ub for _ub in self.ub])
        assert uniformBound, 'The problem bound should be scaled to uniform'
        assert self.n_init < self.max_evals, 'Number of initial points must be smaller than max # evaluations'

        self.lamda_run = []
        
        while self.total_eval < self.max_evals:
            iter_time_start = time.time()            
            self.observed_x = np.zeros((0, self.dim))   # local dataset, reset after restart
            self.observed_fx = np.zeros((0, 1))         # local dataset, reset after restart
            self.reset_info()
 
            # Find initial mean by initial sampling
            stampa=time.time()
            X_init = latin_hypercube(self.n_init, self.dim)
            X_init = from_unit_cube(X_init, self.lb, self.ub)
            stampb=time.time()
            fX_init = np.array([self.f(x) for x in X_init]).reshape(-1, 1)
            self.observed_x = np.vstack((self.observed_x, deepcopy(X_init)))
            self.observed_fx = np.vstack((self.observed_fx, deepcopy(fX_init)))
            self.total_eval += self.n_init
            extra_data = {
                'iter': self.total_eval,
                'running_time': stampb - stampa,
                'x': deepcopy(X_init),
            }
            mean0 = X_init[fX_init.argmin()]
            print(f'{self.total_eval}/{self.max_evals} - fbest: {self.observed_fx.min():.4f}')
           
            domain_length = self.ub[0] - self.lb[0] 
            sigma0 = 0.3*domain_length

            # This will assume to starting covariance matrix to be eye matrix
            self.es = cma.CMAEvolutionStrategy(mean0.flatten(), sigma0, {
                'popsize': self.lamda, 
                'bounds': [self.lb, self.ub],
                'seed': np.nan,
            })

            # Record values
            local_fbest_hist = np.array([self.observed_fx.min()])
            local_fbest = self.observed_fx.min()
            self._storedata(X_init, fX_init)
            iter_time_end = time.time()
            self.lamda_run.append({'iter': self.total_eval, 'total_time': iter_time_end-iter_time_start})
            # Main loop
            while not self.es.stop() and self.total_eval < self.max_evals:     
                time_stamp1 = time.time()
                if self.solver == 'bo':
                    x_fevals, fx_fevals, n_feval, extra_data = self._evaluate_by_bo()
                elif self.solver == 'turbo':
                    x_fevals, fx_fevals, n_feval, extra_data = self._evaluate_by_turbo()
                elif self.solver == 'baxus':
                    x_fevals, fx_fevals, n_feval, extra_data = self._evaluate_by_baxus()
                else:
                    raise NotImplementedError()
                
                self.total_eval += n_feval

                self.observed_x = np.vstack((self.observed_x, deepcopy(x_fevals)))
                self.observed_fx = np.vstack((self.observed_fx, deepcopy(fx_fevals)))
                assert len(self.history_fx) + len(self.observed_fx) == self.total_eval
                self._storedata(x_fevals, fx_fevals, extra_data=extra_data)
                time_stamp2 = time.time()
                if self.baxus_restarted:
                    self.baxus_restart_counter += 1
                    print(f'BAxUS restarted {self.baxus_restart_counter} time(s).')

                if self.turbo_restarted or n_feval == 0:
                    self.history_x = np.vstack((self.history_x, deepcopy(self.observed_x)))
                    self.history_fx = np.vstack((self.history_fx, deepcopy(self.observed_fx)))
                    print(f'sigma/sigma0: {self.es.sigma/(self.es.sigma0):.4f}')
                    print('turbo and cma restarted ...')
                    
                    self.lamda_run.append({
                        'iter': self.total_eval, 
                        'total_time': time_stamp2-time_stamp1, 
                        'restart': True
                        })
                    break

                # Update mean, covariance matrix and step-size
                start = len(fx_fevals) % self.lamda
                end = start + self.lamda
                while end <= len(fx_fevals):
                    self.es.ask()
                    self.es.tell(x_fevals[start:end,:], fx_fevals[start:end,:].flatten())
                    start = end
                    end += self.lamda
                time_stamp3 = time.time()

                # Update fbest
                if len(fx_fevals):
                    if local_fbest > fx_fevals.min():
                        local_fbest = fx_fevals.min()
                local_fbest_hist = np.append(local_fbest_hist, local_fbest)

                print(f'{self.total_eval}/{self.max_evals}({self.func_name})-' +\
                    f' fbest: {np.vstack((self.history_fx, self.observed_fx)).min():.4f}' +\
                    f'; fbest_local: {local_fbest:.4f}' +\
                    f'; sigma/sigma0: {self.es.sigma/(self.es.sigma0):.4f}',
                    f'; tr_length: {self.turbo_prev_length if self.solver=="turbo" else self.baxus_prev_length if self.solver=="baxus" else "None"}'
                )
                time_stamp4 = time.time()
                self.lamda_run.append({
                    'iter': self.total_eval, 
                    'total_time': time_stamp4-time_stamp1, 
                    'cma_update': time_stamp3-time_stamp2, 
                    'restart': False
                    })
                ...

            if self.es.stop():
                print('Stop: ', self.es.stop()) # Print the reason why cmaes package stops
                self.history_x = np.vstack((self.history_x, deepcopy(self.observed_x)))
                self.history_fx = np.vstack((self.history_fx, deepcopy(self.observed_fx)))

        # Append previous data to retrieve all observed data        
        self.observed_x = np.vstack((self.history_x, deepcopy(self.observed_x)))
        self.observed_fx = np.vstack((self.history_fx, deepcopy(self.observed_fx)))
        # end of optimize

    def _initialize_storage(self):
        self.extra_data = []
        self.es_data = []

    def _storedata(self, x_cmapop, fx_cmapop, extra_data=None):
        if self.keep_record:
            self.es_data.append({
                'iter': self.total_eval,
                'x': deepcopy(x_cmapop),
                'fx': deepcopy(fx_cmapop),
                'mean': deepcopy(self.es.mean),
                'sigma': deepcopy(self.es.sigma),
                'cov': deepcopy(self.es.C),
            })
            if isinstance(extra_data, list):
                self.extra_data.extend(deepcopy(extra_data))
            else:
                self.extra_data.append(deepcopy(extra_data))
    
    def _dumpdata(self, seed=0, total_time=None):
        if self.keep_record:
            results = dict()
            results['es_data'] = self.es_data
            results['lb'] = self.lb
            results['ub'] = self.ub
            results['es_mu'] = self.es.sp.mu
            results['f_name'] = self.func_name
            results['dim'] = self.dim
            results['observed_x'] = self.observed_x
            results['observed_fx'] = self.observed_fx
            results['lambda_runtime'] = self.lamda_run
            results['extra_data'] = self.extra_data                
            results['total_time'] = total_time               
            filename = f'cma-{self.solver}_{self.func_name}_{self.dim}D_{seed}'
            with open(filename + '.pkl', 'wb') as f:
                pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)        
        
    def _is_in_ellipse(self, mean, cov, u):
        '''Check whether points is inside a confidence ellipsoid'''
        def mahalanobis_v(d, mean, Sigma):
            Sigma_inv = np.linalg.inv(Sigma)
            xdiff = d - mean
            return np.sqrt(np.einsum('ij,im,mj->i', xdiff, xdiff, Sigma_inv))
        d = mahalanobis_v(u, mean, cov)
        mask = d < self.std
        return mask
          
    def _evaluate_by_bo(self):
        '''Evaluate the population based on a GP updated by BO-TS'''
        # Initialize data for returning       
        es = self.es            
        mask = np.ones(self.observed_fx.shape[0], dtype=bool)           
        x_gp = deepcopy(self.observed_x[mask][-int((4 + math.floor(3*np.log(self.dim)))*100):])
        fx_gp = deepcopy(self.observed_fx[mask][-int((4 + math.floor(3*np.log(self.dim)))*100):])
        
        x_feval = np.zeros((0, self.dim))
        fx_feval = np.zeros((0, 1))
        x_bo = np.zeros((0, self.dim))
        fx_bo = np.zeros((0, 1))

        if len(fx_gp) == 0:
            return x_bo, fx_bo, 0
           
        # Start running BO using Thompson sampling as acquisition function
        n_eval = 0
        bo_data = []
        for cnt in range(self.lamda):
            stamp1 = time.time()
            # Scale x to unit cube and standardize fx for training GP
            x_gp_unit = to_unit_cube(x_gp, self.lb, self.ub)
            mu, sigma = np.median(fx_gp), fx_gp.std()
            sigma = 1.0 if sigma < 1e-6 else sigma
            fx_gp_unit = (deepcopy(fx_gp) - mu) / sigma
            stamp2 = time.time()
            with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                gp = train_gp(torch.tensor(x_gp_unit), torch.tensor(fx_gp_unit.flatten()), True, 50)
            
            stamp3 = time.time()

            # After training GP, let's start to sample TS points
            n_cand = min(100*self.dim, 5000) # number of TS sampling points, formula from TuRBO
            x_cand = np.random.multivariate_normal(es.mean, es.sigma**2*es.C, size=int(n_cand*1.2))
            mask = self._is_in_ellipse(es.mean, es.sigma**2*es.C, x_cand)
            x_cand = x_cand[mask][:n_cand, :]
            
            x_cand = np.array([self.bound_tf.repair(x) for x in x_cand])
            x_cand = to_unit_cube(x_cand, self.lb, self.ub)
            stamp4 = time.time()
            # Start predicting with gpytorch
            with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
                x_torch = torch.tensor(x_cand).to(dtype=torch.float64)
                y_cand = gp.likelihood(gp(x_torch)).sample(torch.Size([1])).t().numpy()

            # Update newly found min value to the training data
            x_min = from_unit_cube(x_cand[y_cand.argmin(axis=0), :], self.lb, self.ub)       
            stamp5 = time.time()
            fx_min = np.array([self.f(x) for x in x_min]).reshape(-1, 1)
            print(f'Iter {self.total_eval + cnt + 1}: {np.around(fx_min.squeeze(), 4)}')
            stamp6 = time.time()
            n_eval += 1
            x_gp = np.vstack((x_gp, x_min))
            fx_gp = np.vstack((fx_gp, deepcopy(fx_min)))
            x_bo = np.vstack((x_bo, x_min))
            fx_bo = np.vstack((fx_bo, fx_min))
            del gp, x_cand, y_cand
            stamp7 = time.time()
            bo_data.append({
                'iter': n_eval+self.total_eval,
                'running_time':{
                    'iter_run': (stamp7 - stamp6) + (stamp5 - stamp1),
                    'gp_train': stamp3 - stamp2,
                    'sampling': stamp4 - stamp3,
                    'prediction': stamp5 - stamp4,
                },
                'x': deepcopy(x_min),
            })

        return np.vstack((x_feval, x_bo)), np.vstack((fx_feval, fx_bo)), n_eval, bo_data
            
    def _evaluate_by_turbo(self):
        '''Evaluate the population based on a GP updated by TuRBO'''
       
        mask = np.ones(self.observed_fx.shape[0], dtype=bool)
              
        x_turbo_init = deepcopy(self.observed_x[mask])
        fx_turbo_init = deepcopy(self.observed_fx[mask])

        if len(fx_turbo_init) == 0:
            self.turbo_restarted = True
            x_bo = np.zeros((0, self.dim))
            fx_bo = np.zeros((0, 1))
        else:      
            def create_candidates(n_cand, length):
                mean = self.es.mean
                cov = self.es.sigma**2*self.es.C
                eival, eivec = np.linalg.eigh(cov)
                eival = np.sqrt(eival)

                # trust region
                new_eigval = np.square(eival*length)
                new_cov = eivec @ np.diag(new_eigval) @ np.linalg.inv(eivec)
                
                # cma ellipse
                x_cand = np.random.multivariate_normal(mean, new_cov, size=int(1.2*n_cand))
                x_cand = x_cand[self._is_in_ellipse(mean, new_cov, x_cand)][:n_cand,:]
                x_cand = np.array([self.bound_tf.repair(x) for x in x_cand])
                x_cand_unit = to_unit_cube(x_cand, self.lb, self.ub)  
                return x_cand_unit
            
            turbo = Turbo1(
                f=self.f,               # Handle to objective function
                lb=self.lb,             # Numpy array specifying lower bounds
                ub=self.ub,             # Numpy array specifying upper bounds
                x_init=x_turbo_init,    # Number of initial bounds from an Latin hypercube design
                fx_init=fx_turbo_init,  # Number of initial bounds from an Latin hypercube design
                max_evals=self.lamda,    # Maximum number of evaluations
                batch_size=1,           # How large batch size TuRBO uses
                verbose=False,          # Print information from each batch
                use_ard=True,           # Set to true if you want to use ARD for the GP kernel
                max_cholesky_size=self.max_cholesky_size, # When we switch from Cholesky to Lanczos
                n_training_steps=50,    # Number of steps of ADAM to learn the hypers
                min_cuda=1024,          # Run on the CPU for small datasets
                device="cpu",           # "cpu" or "cuda"
                dtype="float64",        # float64 or float32
                n_eval_offset=self.total_eval,
                cma_succcount=self.turbo_succcount,
                cma_failcount=self.turbo_failcount,
                prev_length=self.turbo_prev_length,
                create_candidates=create_candidates,
                turbo_gp_init_x=self.turbo_gp_init_x,
                turbo_gp_init_fx=self.turbo_gp_init_fx,
            )
        
            self.turbo_restarted = False
            turbo.optimize()
            x_bo = turbo.X_eval
            fx_bo = turbo.fX_eval
            self.turbo_prev_length = turbo.length
            self.turbo_succcount = turbo.succcount
            self.turbo_failcount = turbo.failcount
            self.turbo_restarted = turbo.restarted
            self.turbo_gp_init_x = turbo.turbo_gp_init_x
            self.turbo_gp_init_fx = turbo.turbo_gp_init_fx
            turbo_data = turbo.turbo_data

            if len(fx_bo) < self.lamda:
                self.turbo_restarted = True

        if self.turbo_restarted:
            self.turbo_prev_length = 0.8
            self.turbo_succcount = 0
            self.turbo_failcount = 0

        return x_bo, fx_bo, len(fx_bo), turbo_data
    
    def _evaluate_by_baxus(self):
        class CustomFunction(Benchmark):
            def __init__(self, f, dim, lb, ub):
                super().__init__(dim=dim, ub=ub, lb=lb, noise_std=0)
                self.objective = f

            def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
                fx = float(self.objective(x))
                return fx

        if not os.path.isdir('cma-baxus_results'):
            os.makedirs('cma-baxus_results')

 
        currentDT = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') 

        custom_function = CustomFunction(f=self.f, dim=self.dim, lb=self.lb, ub=self.ub)

        n_init = self.lamda - 1 # in the beginnig, use the same initialization as baxus
        baxus_data_fx = []
        baxus_gp_data_x = []
        if len(self.baxus_embedded_x) > 0:
            n_init = 0
            baxus_gp_data_x = np.array([x for x, _ in self.baxus_embedded_x])
            baxus_global_X = np.array([x for _, x in self.baxus_embedded_x])
            baxus_data_fx = self.baxus_embedded_fx
        adjust_initial_target_dim = (len(self.baxus_embedded_x) == 0) and (self.baxus_prev_target_dim < self.dim)
        behavior = BaxusBehavior(
            adjust_initial_target_dim=adjust_initial_target_dim,
            budget_until_input_dim=self.max_evals,
            )
        
        def create_candidates(n_cand, S, length):
            # mvn in X: [self.lb, self.ub]^D
            mean_X = self.es.mean
            cov_X = self.es.sigma**2 * self.es.C           

            # scale the mvn to [-1, 1]^D
            scale_mtrx = np.diag((2/(self.ub - self.lb)))
            mean_X_scaled = scale_mtrx @ (mean_X - (self.ub + self.lb)/2.0)
            cov_X_scaled = scale_mtrx @ cov_X @ scale_mtrx.T
            eival, eivec = np.linalg.eigh(cov_X_scaled)
            eival = np.sqrt(eival)
            new_eigval = np.square(eival*length)
            cov_X_scaled_tr = eivec @ np.diag(new_eigval) @ np.linalg.inv(eivec)
            test = to_1_around_origin(mean_X.reshape(1, -1), self.lb, self.ub).flatten()
            assert np.allclose(test,mean_X_scaled)

            # project mvn to Y: [-1, 1]^d
            SST_torch = torch.tensor(S @ S.T) + 1e-6*np.eye(S.shape[0])
            S_torch = torch.tensor(S)
            P = (torch.linalg.inv(SST_torch) @ S_torch).detach().numpy()
            mean_Y = P @ mean_X_scaled
            cov_Y = P @ cov_X_scaled_tr @ P.T

            # start sampling
            X_cand = np.random.multivariate_normal(mean_Y, cov_Y, size=int(1.2*n_cand))
            X_cand = X_cand[self._is_in_ellipse(mean_Y, cov_Y, X_cand)][:n_cand,:]
            bound_tf = cma.BoundTransform([-1., 1.])
            X_cand = np.array([bound_tf.repair(x) for x in X_cand])

            return X_cand, -1.0, 1.0
    

        baxus = BAxUS(
            run_dir=f'cma-baxus_results/{str(currentDT)}',
            max_evals=self.lamda,
            n_init=n_init,
            f=custom_function,
            target_dim=self.baxus_prev_target_dim,
            verbose=True,
            max_cholesky_size=self.max_cholesky_size,
            offset=self.total_eval,
            behavior=behavior,
            create_cand_cma=create_candidates,
        )

        if len(baxus_gp_data_x) > 0:
            baxus.extra_x_gp = deepcopy(baxus_gp_data_x)
            baxus.extra_fx = deepcopy(baxus_data_fx)
            baxus.extra_x = deepcopy(baxus_global_X)
            baxus.lengthscales = self.baxus_prev_lengthscales
            baxus.prev_length = self.baxus_prev_length
            baxus.prev_succcount = self.baxus_succcount
            baxus.prev_failcount = self.baxus_failcount
            baxus.projector =  self.baxus_projector
            baxus._init_target_dim = self.baxus_init_target_dim
        baxus._trust_region_restarts = self.baxus_trust_region_restarts
        baxus._dim_in_iterations = self.baxus_dim_in_iterations
        baxus._axus_change_iterations = self.baxus_axus_change_iterations
        baxus._split_points = self.baxus_split_points

        logging.root.setLevel(logging.DEBUG)
        baxus.optimize()
        logging.root.setLevel(logging.NOTSET)

        x_raw, y_raw = baxus.optimization_results_raw()  # get the points in the search space and their function values
        init_len = len(baxus_data_fx)
        x_baxus_all = np.clip(np.array(x_raw), self.lb, self.ub) 
        y_baxus_all = np.array(y_raw).reshape(-1, 1)
        x_baxus = x_baxus_all[init_len:, :]
        y_baxus = y_baxus_all[init_len:, :]
        
        self.baxus_prev_length = baxus.length
        self.baxus_succcount = baxus.succcount
        self.baxus_failcount = baxus.failcount
        self.baxus_prev_target_dim = int(baxus._target_dim)
        self.baxus_restarted = baxus.terminated
        self.baxus_prev_lengthscales = baxus.lengthscales
        self.baxus_projector = baxus.projector
        self.baxus_axus_change_iterations = baxus._axus_change_iterations
        self.baxus_split_points = baxus._split_points
        self.baxus_trust_region_restarts = baxus._trust_region_restarts
        self.baxus_dim_in_iterations = baxus._dim_in_iterations
        self.baxus_init_target_dim = baxus._init_target_dim
        baxus_data = baxus.baxus_data

        assert len(baxus._X) == len(x_baxus_all)
        if baxus.terminated:
            self.baxus_embedded_x = []
            self.baxus_embedded_fx = []
        else:
            self.baxus_embedded_x = [(embedded, original) for embedded, original in zip(baxus._X, x_baxus_all)]
            self.baxus_embedded_fx = y_baxus_all


        del baxus
        return x_baxus, y_baxus, len(y_baxus), baxus_data
