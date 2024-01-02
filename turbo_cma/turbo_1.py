
import math
import sys
import time
from copy import deepcopy

import gpytorch
import numpy as np
import torch

from cmabo.gp import train_gp
from .utils import from_unit_cube, to_unit_cube


class Turbo1:
    """The TuRBO-1 algorithm.

    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
    n_init : Number of initial points (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")

    Example usage:
        turbo1 = Turbo1(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals)
        turbo1.optimize()  # Run optimization
        X, fX = turbo1.X, turbo1.fX  # Evaluated points
    """

    def __init__(
        self,
        f,
        lb,
        ub,
        x_init,
        fx_init,
        max_evals,
        batch_size=1,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
        cma_succcount=0,
        cma_failcount=0,
        prev_length=0.8,
        n_eval_offset=0,
        create_candidates=None,
        turbo_gp_init_x=None,
        turbo_gp_init_fx=None,
    ):

        # Very basic input checks
        assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub)
        assert np.all(ub > lb)
        assert max_evals > 0 and isinstance(max_evals, int)
        # assert isinstance(x_init, np.array)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert isinstance(verbose, bool) and isinstance(use_ard, bool)
        assert max_cholesky_size >= 0 and isinstance(batch_size, int)
        assert n_training_steps >= 30 and isinstance(n_training_steps, int)
        assert max_evals > batch_size
        assert device == "cpu" or device == "cuda"
        assert dtype == "float32" or dtype == "float64"
        if device == "cuda":
            assert torch.cuda.is_available(), "can't use cuda if it's not available"

        # Save function information
        self.f = f
        self.dim = len(lb)
        self.lb = lb
        self.ub = ub

        # Settings
        self.x_init = x_init
        self.fx_init = fx_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps

        # Hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = np.zeros((0, self.dim)) if self.use_ard else np.zeros((0, 1))

        # Tolerances and counters
        self.n_cand = min(100 * self.dim, 5000)
        self.failtol = np.ceil(np.max([4.0 / batch_size, self.dim / batch_size]))
        self.succtol = 3
        self.n_evals = 0

        # Trust region sizes
        self.length_min = 0.5 ** 7
        self.length_max = 1.6
        self.length_init = 0.8

        # Save the full history (without initial points)
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))

        # Save the full history (without initial points)
        self.X_eval = np.zeros((0, self.dim))
        self.fX_eval = np.zeros((0, 1))

        # Device and dtype for GPyTorch
        self.min_cuda = min_cuda
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")

        # Information from CMA
        self.cma_succcount = cma_succcount # previous turbo information
        self.cma_failcount = cma_failcount # previous turbo information
        self.prev_length = prev_length # previous turbo information
        self.restarted = False # has turbo restarted?
        self.create_candidates_cma = create_candidates # TS candidates generation
        self.turbo_gp_init_x = turbo_gp_init_x # previous data for building GP
        self.turbo_gp_init_fx = turbo_gp_init_fx # previous data for building GP
        self.turbo_data = [] # for logging
        self.n_eval_offset = n_eval_offset # for offset the printing

        # Initialize parameters
        self._restart()

    def _restart(self):
        self._X = []
        self._fX = []
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init #*np.ones(self.dim)

    def _adjust_length(self, fX_next):
        if np.min(fX_next) < np.min(self._fX) - 1e-3 * math.fabs(np.min(self._fX)):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1
        print(f'Iter {self.n_eval_offset + self.n_evals + 1}: {np.around(fX_next.squeeze(), 4)}; succount = {self.succcount}/{self.succtol}; failcount = {self.failcount}/{self.failtol}; length = {self.length}')

        if self.succcount == self.succtol:  # Expand trust region
            self.length = min([2.0 * self.length, self.length_max])
            # self.length = np.clip(2.0*self.length, a_min=None, a_max=self.length_max)
            self.succcount = 0
        elif self.failcount == self.failtol:  # Shrink trust region
            self.length /= 2.0
            self.failcount = 0

    def _create_candidates(self, X, fX, length, n_training_steps, hypers):
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        assert X.min() >= 0.0 and X.max() <= 1.0

        # Standardize function values.
        mu, sigma = np.median(fX), fX.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        fX = (deepcopy(fX) - mu) / sigma

        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        stamp2 = time.time()
        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            gp = train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers
            )

            # Save state dict
            hypers = gp.state_dict()
        stamp3 = time.time()

        # Create the trust region boundaries
        X_cand = self.create_candidates_cma(self.n_cand, length)
        stamp4 = time.time()    
        if (len(X_cand) == 0):
            print(f'\033[93m ---->len(X_cand) == 0<----\033[0m')
            raise ValueError('len(X_cand) == 0')

        # Figure out what device we are running on
        if len(X_cand) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We may have to move the GP to a new device
        gp = gp.to(dtype=dtype, device=device)

        # We use Lanczos for sampling if we have enough data
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
            y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()

        # Remove the torch variables
        del X_torch, y_torch, X_cand_torch, gp

        # De-standardize the sampled values
        y_cand = mu + sigma * y_cand

        return X_cand, y_cand, hypers, (stamp2, stamp3, stamp4)

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates."""
        X_next = np.ones((self.batch_size, self.dim))
        for i in range(self.batch_size):
            # Pick the best point and make sure we never pick it again
            indbest = np.argmin(y_cand[:, i])
            X_next[i, :] = deepcopy(X_cand[indbest, :])
            y_cand[indbest, :] = np.inf
        return X_next

    def optimize(self):
        """Run optimization process."""
        # Initialize parameters
        self._restart()
        if self.prev_length >= self.length_min:
            # set information from previous turbo run
            self.succcount = self.cma_succcount
            self.failcount = self.cma_failcount
            self.length = min(self.prev_length, self.length_max)
        else:        
            return
            
        X_init = self.x_init
        fX_init = self.fx_init
        # Update budget and set as initial data for this TR
        self._X = deepcopy(X_init)
        self._fX = deepcopy(fX_init)
        
        # Append data to the global history
        self.X = np.vstack((self.X, deepcopy(X_init)))
        self.fX = np.vstack((self.fX, deepcopy(fX_init)))
        

        if self.verbose:
            fbest = self._fX.min()
            print(f"Starting from fbest = {fbest:.4}")
            sys.stdout.flush()

        # Thompson sample to get next suggestions
        while self.n_evals < self.max_evals and self.length >= self.length_min:
            stamp1 = time.time()
            # Warp inputs
            X = to_unit_cube(deepcopy(self._X), self.lb, self.ub)

            # Standardize values
            fX = deepcopy(self._fX).ravel()

            # Create th next batch
            X_cand, y_cand, _, (stamp2, stamp3, stamp4) = self._create_candidates(
                X, fX, length=self.length, n_training_steps=self.n_training_steps, hypers={}
            )
            X_next = self._select_candidates(X_cand, y_cand)

            # Undo the warping
            X_next = from_unit_cube(X_next, self.lb, self.ub)
            
            stamp5 = time.time()
            # Evaluate batch
            fX_next = np.array([self.f(x) for x in X_next]).reshape(-1, 1)
            stamp6 = time.time()

            # Update trust region
            self._adjust_length(fX_next)

            # Update budget and append data
            self.n_evals += self.batch_size
            self._X = np.vstack((self._X, X_next))
            self._fX = np.vstack((self._fX, fX_next))

            if self.verbose and fX_next.min() < self.fX.min():
                n_evals, fbest = self.n_evals, fX_next.min()
                print(f"{self.n_eval_offset+n_evals}) New best: {fbest:.4}")
                sys.stdout.flush()

            # Append data to the global history
            self.X = np.vstack((self.X, deepcopy(X_next)))
            self.fX = np.vstack((self.fX, deepcopy(fX_next)))

            self.X_eval = np.vstack((self.X_eval, deepcopy(X_next)))
            self.fX_eval = np.vstack((self.fX_eval, deepcopy(fX_next)))

            self.turbo_gp_init_x = np.vstack((self.turbo_gp_init_x, deepcopy(X_next)))
            self.turbo_gp_init_fx = np.vstack((self.turbo_gp_init_fx, deepcopy(fX_next)))
            stamp7 = time.time()

            self.turbo_data.append({
                'iter': self.n_eval_offset + self.n_evals,
                'running_time': {
                    'iter_run': (stamp7 - stamp6) + (stamp5 - stamp1),
                    'gp_train': stamp3 - stamp2,
                    'sampling': stamp4 - stamp3,
                    'prediction': stamp5 - stamp4,
                },
                'x': deepcopy(X_next),
                'tr': {
                    'succount': self.succcount,
                    'failcount': self.failcount,
                    'length': self.length
                }
            })
        # if any(self.length < self.length_min):
        #     self.restarted = True