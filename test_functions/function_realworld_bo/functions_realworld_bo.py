# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 23:24:07 2023

@author: 
@description: Some high dim realworld functions for testing Bayesian Optimization
"""
import os
import stat
import subprocess
import sys
import tempfile
import urllib
from collections import OrderedDict
from logging import info, warning
from platform import machine
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from .bipedal_walker import BipedalWalker, heuristic_bipedal
from .ebo_core.helper import ConstantOffsetFn, NormalizedInputFn
from .lunar_lander import LunarLander, heuristic_turbo
from .push_function import PushReward
from .rover_function import create_large_domain, create_small_domain, create_large_domain_50


class Bipedal_walking:
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 13

        if bounds is None:
            self.bounds = [(-1., 1.)]*self.input_dim
        else:
            self.bounds = bounds

        # self.min = [(0.)*self.input_dim]
        # self.fmin = 1
        # self.ismax = 1
        self.name = 'bipedal-walking'
        
        # env = BipedalWalker()
        # s_all = []
        # N = 50
        # for i in range(N):
        #     seed = i
        #     # env.seed(seed)
        #     s = env.reset()
        #     s_all.append(s)
        # self.s_all = s_all
    
    def func(self, x):
        assert(all(x <= 1.) and all(x >= -1.))
        reward_all = []
        for i in range(5):
            env = BipedalWalker()
            s = env.reset()
            np.random.seed(i)
            total_reward = 0
            steps = 0
            while True & (steps <= 2000):
                a = heuristic_bipedal(s, x)
                s, r, done, info = env.step(a)
                total_reward += r
                steps += 1
                if done: break
        
            reward_all.append(total_reward)
        value = np.mean(reward_all)
        
        return -value


class Lunar_landing:
    
    '''
    Lunar landing - from OpenAI gym
    '''
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 12

        if bounds is None:
            self.bounds = [(0.0, 2.0)]*self.input_dim
        else:
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = 1
        self.ismax = 1
        self.name = 'lunar-landing'
        
        env = LunarLander()
        s_all = []
        N = 50
        for i in range(N):
            seed = i
            env.seed(seed)
            s = env.reset()
            s_all.append(s)
        self.s_all = s_all
    
    def func(self, x):

        reward_all = []
        for i, s in enumerate(self.s_all):
            env = LunarLander()
            env.seed(i)
            np.random.seed(i)
            total_reward = 0
            steps = 0
            while True & (steps <= 1000):
                a = heuristic_turbo(s, x)
                s, r, done, info = env.step(a)
                total_reward += r
                steps += 1
                if done: break
        
            reward_all.append(total_reward)
        value = np.mean(reward_all)
        
        return -value


class Robot_pushing:
    
    '''
    Robot pushing - from the Github: https://github.com/zi-w/Ensemble-Bayesian-Optimization
    '''
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 14

        if bounds is None:
            
            # self.bounds = OrderedDict([('x1', (-5, 5)),
            #                             ('x2', (-5, 5)),
            #                             ('x3', (-10, 10)),
            #                             ('x4', (-10, 10)),
            #                             ('x5', (2, 30)),
            #                             ('x6', (0, 2*np.pi)),
            #                             ('x7', (-5, 5)),
            #                             ('x8', (-5, 5)),
            #                             ('x9', (-10, 10)),
            #                             ('x10', (-10, 10)),
            #                             ('x11', (2, 30)),
            #                             ('x12', (0, 2*np.pi)),
            #                             ('x13', (-5, 5)),
            #                             ('x14', (-5, 5))])

            self.bounds = OrderedDict([('x1', (-5.0, 5.0)),
                                        ('x2', (-5.0, 5.0)),
                                        ('x3', (-5.0, 5.0)),
                                        ('x4', (-5.0, 5.0)),
                                        ('x5', (-5.0, 5.0)),
                                        ('x6', (-5.0, 5.0)),
                                        ('x7', (-5.0, 5.0)),
                                        ('x8', (-5.0, 5.0)),
                                        ('x9', (-5.0, 5.0)),
                                        ('x10', (-5.0, 5.0)),
                                        ('x11', (-5.0, 5.0)),
                                        ('x12', (-5.0, 5.0)),
                                        ('x13', (-5.0, 5.0)),
                                        ('x14', (-5.0, 5.0))])
        else:
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = 1
        self.ismax = 1
        self.name = 'robot-pushing'
    
    def func(self, x):
        
        # convert the value of x
        x_transform = x.copy()
        
        x_transform[2] = 2*x_transform[2]
        x_transform[3] = 2*x_transform[3]
        x_transform[4] = 2.8*x_transform[4] + 16
        x_transform[5] = (2*np.pi -4)/10*x_transform[5] + np.pi
        x_transform[8] = 2*x_transform[8]
        x_transform[9] = 2*x_transform[9]
        x_transform[10] = 2.8*x_transform[10] + 16
        x_transform[11] = (2*np.pi -4)/10*x_transform[11] + np.pi
        
        f = PushReward()
        x_transform = x_transform.ravel()
        fval = f(x_transform)
        
        return -fval


class Rover:
    '''
    Rover - from Turbo
    '''
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 60

        if bounds is None:
            self.bounds = [(0., 1.)]*self.input_dim
        else:
            self.bounds = bounds

        self.fmax = 5
        self.ismax = 1
        self.name = 'rover'
    
    def func(self, x):
        def l2cost(x, point):
            return 10 * np.linalg.norm(x - point, 1)
        # domain = create_small_domain()
        domain = create_large_domain(force_start=False,
                                    force_goal=False,
                                    start_miss_cost=l2cost,
                                    goal_miss_cost=l2cost)
        n_points = domain.traj.npoints
        
        raw_x_range = np.repeat(domain.s_range, n_points, axis=1)
        
        # maximum value of f
        f_max = 5.0
        f = ConstantOffsetFn(domain, f_max)
        f = NormalizedInputFn(f, raw_x_range)
        
        x = x.ravel()
        fval = f(x)

        return -fval


class Rover20:
    '''
    Rover - from Turbo
    '''
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 20

        if bounds is None:
            self.bounds = [(0., 1.)]*self.input_dim
        else:
            self.bounds = bounds

        self.fmax = 5
        self.ismax = 1
        self.name = 'rover'
    
    def func(self, x):
        # def l2cost(x, point):
        #     return 10 * np.linalg.norm(x - point, 1)
        domain = create_small_domain()
        # domain = create_large_domain(force_start=False,
        #                             force_goal=False,
        #                             start_miss_cost=l2cost,
        #                             goal_miss_cost=l2cost)
        n_points = domain.traj.npoints
        
        raw_x_range = np.repeat(domain.s_range, n_points, axis=1)
        
        # maximum value of f
        f_max = 5.0
        f = ConstantOffsetFn(domain, f_max)
        f = NormalizedInputFn(f, raw_x_range)
        
        x = x.ravel()
        fval = f(x)

        return -fval


class Rover100:
    '''
    Rover - from Turbo
    '''
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 100

        if bounds is None:
            self.bounds = [(0., 1.)]*self.input_dim
        else:
            self.bounds = bounds

        self.fmax = 5
        self.ismax = 1
        self.name = 'rover'
    
    def func(self, x):
        def l2cost(x, point):
            return 10 * np.linalg.norm(x - point, 1)
        # domain = create_small_domain()
        domain = create_large_domain_50(force_start=False,
                                    force_goal=False,
                                    start_miss_cost=l2cost,
                                    goal_miss_cost=l2cost)
        n_points = domain.traj.npoints
        
        raw_x_range = np.repeat(domain.s_range, n_points, axis=1)
        
        # maximum value of f
        f_max = 5.0
        f = ConstantOffsetFn(domain, f_max)
        f = NormalizedInputFn(f, raw_x_range)
        
        x = x.ravel()
        fval = f(x)

        return -fval
    
class ElectronSphere6np:
    def __init__(self):
        self.n_p = int(6)
        self.high_dim = self.n_p * int(2)
        self.input_dim = self.high_dim
        
        self.bounds = [(0, 1)]*self.input_dim

        self.fmin = 9.985281
        self.minimizer = np.array([[0.48682843, 0.78674212, 0.57885328, 0.31341441, 0.29749929, 0.40868617, 0.07889095,
                                    0.68647393, 0.98678081, 0.2134249, 0.79744743, 0.59130902]])
        self.scale = 0.01
        self.name = 'electron6'

    def spherical_to_cartesian(self, alpha_reshape):
        '''
        Convert psherical coordinates to cartesian ones: (\theta, \phi) -> (x, y, z)
        \theta: Azimuthal angle (interval: [0, 2*\pi])
        \phi:   Polar angle     (interval: [0, \pi])
        Radious fixed to 1 (all points are in a unit sphere)
        :param alpha: Spherical coordinates (N x self.high_dim): [N x (\theta_0, \phi_0, \theta_1, \phi_1, ... ,\theta_np, \phi_np)]
        :return:  Cartesian Coordinates x (N x self.n_p), y (N x self.n_p), z (N x self.n_p)
        '''

        if np.all(alpha_reshape[:, :, 0] <= 2 * np.pi) and np.all(alpha_reshape[:, :, 1] <= np.pi) and np.all(
            alpha_reshape >= 0.):
            # print('Input ElectronSphere6np as expected')
            pass
        else:
            # print(alpha_reshape)
            pass
        x = np.multiply(np.cos(alpha_reshape[:, :, 0]), np.sin(alpha_reshape[:, :, 1]))     # N x self.n_p
        y = np.multiply(np.sin(alpha_reshape[:, :, 0]), np.sin(alpha_reshape[:, :, 1]))     # N x self.n_p
        z = np.cos(alpha_reshape[:, :, 1])                                                  # N x self.n_p
        return x, y, z

    def func(self, x, noisy=False, fulldim=False):
        '''
        Assuming the input x is (N x self.high_dim), with values in the range [0, 1]
        :param x:
        :return:
        '''
        if len(x.shape) == 1:
            x = x[None]

        # structure according to assumption: [N x (\alpha_0, \beta_0, \alpha_1, \beta_1, ... ,\alpha_np, \beta_np)]
        x_reshape = np.reshape(x, newshape=[np.shape(x)[0], self.n_p, int(2)])  
        theta = np.copy(x_reshape[:, :, 0] * 2 * np.pi)    
        phi = np.copy(x_reshape[:, :, 1] * np.pi)          
        spherical = np.stack([theta, phi], axis=-1)
        x, y, z = self.spherical_to_cartesian(spherical)

        x_Mat = x[:, :, None] - x[:, None, :]
        y_Mat = y[:, :, None] - y[:, None, :]
        z_Mat = z[:, :, None] - z[:, None, :]

        x_Mat2 = x_Mat ** 2.0
        y_Mat2 = y_Mat ** 2.0
        z_Mat2 = z_Mat ** 2.0

        Mat2 = (x_Mat2 + y_Mat2 + z_Mat2) ** (- 0.5)

        iu = np.triu_indices(self.n_p, k=1)
        upper_sum = []
        for mat2_i in list(Mat2):
            upper_sum.append(np.sum(mat2_i[iu]))
        f_x = np.stack(upper_sum, axis=0)[:, None]
        if f_x.max() == np.inf or f_x.max() >= 1e09:
            f_x[np.logical_or(f_x[:, 0] == np.inf, f_x[:, 0] >= 1e09), :] = 1e09
        if noisy:
            noise = np.array([np.random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x   


class ElectronSphere9np:
    def __init__(self):
        self.n_p = int(9)
        self.high_dim = self.n_p * int(2)
        self.input_dim = self.high_dim

        self.bounds = [(0, 1)]*self.input_dim

        self.fmin = 25.759987
        self.minimizer = np.array([[0.5965926, 0.53901935, 0.7879406, 0.49276188, 0.20936081, 0.33062902, 0.21399158,
                                    0.71518286, 0.99539223, 0.60927018, 0.95319699, 0.23248716, 0.70285202, 0.90039541,
                                    0.53554447, 0.1667014, 0.40301371, 0.53355599]])
        self.scale = 0.01
        self.name = 'electron9'

    def spherical_to_cartesian(self, alpha_reshape):
        '''
        Convert psherical coordinates to cartesian ones: (\theta, \phi) -> (x, y, z)
        \theta: Azimuthal angle (interval: [0, 2*\pi])
        \phi:   Polar angle     (interval: [0, \pi])
        Radious fixed to 1 (all points are in a unit sphere)
        :param alpha: Spherical coordinates (N x self.high_dim): [N x (\theta_0, \phi_0, \theta_1, \phi_1, ... ,\theta_np, \phi_np)]
        :return:  Cartesian Coordinates x (N x self.n_p), y (N x self.n_p), z (N x self.n_p)
        '''
        assert np.all(alpha_reshape[:, :, 0] <= 2 * np.pi) and np.all(alpha_reshape[:, :, 1] <= np.pi) and np.all(
            alpha_reshape >= 0.)
        x = np.multiply(np.cos(alpha_reshape[:, :, 0]), np.sin(alpha_reshape[:, :, 1]))     # N x self.n_p
        y = np.multiply(np.sin(alpha_reshape[:, :, 0]), np.sin(alpha_reshape[:, :, 1]))     # N x self.n_p
        z = np.cos(alpha_reshape[:, :, 1])                                                  # N x self.n_p
        return x, y, z

    def func(self, x, noisy=False, fulldim=False):
        '''
        Assuming the input x is (N x self.high_dim), with values in the range [0, 1]
        :param x:
        :return:
        '''
        if len(x.shape) == 1:
            x = x[None]

        # structure according to assumption: [N x (\alpha_0, \beta_0, \alpha_1, \beta_1, ... ,\alpha_np, \beta_np)]
        x_reshape = np.reshape(x, newshape=[np.shape(x)[0], self.n_p, int(2)])
        theta = np.copy(x_reshape[:, :, 0] * 2 * np.pi)
        phi = np.copy(x_reshape[:, :, 1] * np.pi)
        spherical = np.stack([theta, phi], axis=-1)
        x, y, z = self.spherical_to_cartesian(spherical)


        x_Mat = x[:, :, None] - x[:, None, :]
        y_Mat = y[:, :, None] - y[:, None, :]
        z_Mat = z[:, :, None] - z[:, None, :]

        x_Mat2 = x_Mat ** 2.0
        y_Mat2 = y_Mat ** 2.0
        z_Mat2 = z_Mat ** 2.0

        Mat2 = (x_Mat2 + y_Mat2 + z_Mat2) ** (- 0.5)

        iu = np.triu_indices(self.n_p, k=1)
        upper_sum = []
        for mat2_i in list(Mat2):
            upper_sum.append(np.sum(mat2_i[iu]))
        f_x = np.stack(upper_sum, axis=0)[:, None]
        if f_x.max() == np.inf:
            f_x[f_x[:, 0] == np.inf, :] = 1e09
        if noisy:
            noise = np.array([np.random.gauss(0, self.scale) for _ in range(f_x.shape[0])])[:, None]
            y = f_x + noise
            return y
        return f_x    # Nx1


class MoptaSoftConstraints:
    """
    Mopta08 benchmark with soft constraints as described in https://arxiv.org/pdf/2103.00349.pdf
    Supports i386, x86_84, armv7l

    Args:
        temp_dir: Optional[str]: directory to which to write the input and output files (if not specified, a temporary directory will be created automatically)
        binary_path: Optional[str]: path to the binary, if not specified, the default path will be used
    """

    def __init__(
            self,
            temp_dir: Optional[str] = None,
            binary_path: Optional[str] = None,
            noise_std: Optional[float] = 0,
            **kwargs,
    ):
        # super().__init__(124, np.ones(124), np.zeros(124), noise_std=noise_std)
        lb = np.zeros(124)
        ub = np.ones(124)
        self.noise_std = noise_std
        self._dim = 124
        self._lb_vec = lb.astype(np.float32)
        self._ub_vec = ub.astype(np.float32)
        if binary_path is None:
            self.sysarch = 64 if sys.maxsize > 2 ** 32 else 32
            self.machine = machine().lower()
            if self.machine == "armv7l":
                assert self.sysarch == 32, "Not supported"
                self._mopta_exectutable = "mopta08_armhf.bin"
            elif self.machine == "x86_64":
                assert self.sysarch == 64, "Not supported"
                self._mopta_exectutable = "mopta08_elf64.bin"
            elif self.machine == "i386":
                assert self.sysarch == 32, "Not supported"
                self._mopta_exectutable = "mopta08_elf32.bin"
            elif self.machine == "amd64":
                assert self.sysarch == 64, "Not supported"
                self._mopta_exectutable = "mopta08_amd64.exe"
            else:
                raise RuntimeError("Machine with this architecture is not supported")
            self._mopta_exectutable = os.path.join(
                os.getcwd(), "test_functions", "function_realworld_bo", "mopta08", self._mopta_exectutable
            )

            if not os.path.exists(self._mopta_exectutable):
                basename = os.path.basename(self._mopta_exectutable)
                print(f"Mopta08 executable for this architecture not locally available. Downloading '{basename}'...")
                urllib.request.urlretrieve(
                    f"https://mopta.papenmeier.io/{os.path.basename(self._mopta_exectutable)}",
                    self._mopta_exectutable)
                os.chmod(self._mopta_exectutable, stat.S_IXUSR)

        else:
            self._mopta_exectutable = binary_path
        if temp_dir is None:
            self.directory_file_descriptor = tempfile.TemporaryDirectory()
            self.directory_name = self.directory_file_descriptor.name
        else:
            if not os.path.exists(temp_dir):
                warning(f"Given directory '{temp_dir}' does not exist. Creating...")
                os.mkdir(temp_dir)
            self.directory_name = temp_dir
        
        # custom param
        self.input_dim = self._dim
        self.name = 'mopta08'
        self.bounds = [(0., 1.)]*self._dim

    def __call__(self, x):
        # super(MoptaSoftConstraints, self).__call__(x)
        x = np.array(x)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        # create tmp dir for mopta binary

        vals = np.array([self._call(y) for y in x]).squeeze()
        return vals + np.random.normal(
            np.zeros_like(vals), np.ones_like(vals) * self.noise_std, vals.shape
        )

    def _call(self, x: np.ndarray):
        """
        Evaluate Mopta08 benchmark for one point

        Args:
            x: one input configuration

        Returns:value with soft constraints

        """
        assert x.ndim == 1
        # write input to file in dir
        with open(os.path.join(self.directory_name, "input.txt"), "w+") as tmp_file:
            for _x in x:
                tmp_file.write(f"{_x}\n")
        # pass directory as working directory to process
        popen = subprocess.Popen(
            self._mopta_exectutable,
            stdout=subprocess.PIPE,
            cwd=self.directory_name,
        )
        popen.wait()
        # read and parse output file
        output = (
            open(os.path.join(self.directory_name, "output.txt"), "r")
            .read()
            .split("\n")
        )
        output = [x.strip() for x in output]
        output = np.array([float(x) for x in output if len(x) > 0])
        value = output[0]
        constraints = output[1:]
        # see https://arxiv.org/pdf/2103.00349.pdf E.7
        return value + 10 * np.sum(np.clip(constraints, a_min=0, a_max=None))
    
    def func(self, x):
        return self.__call__(x)
    

class SVMBenchmark():
    def __init__(
            self,
            data_folder: Optional[str] = None,
            noise_std: Optional[float] = 0,
            **kwargs,
    ):
        """
        SVM Benchmark from https://arxiv.org/abs/2103.00349

        Support also a noisy version where the model is trained on random subset of 250 points
        which is used whenever noise_std is greater than 0.

        Args:
            data_folder: the folder where the slice_localization_data.csv is located
            noise_std: noise standard deviation. Anything greater than 0 will lead to a noisy benchmark
            **kwargs:
        """
        lb = np.zeros(388)
        ub = np.ones(388)
        self.noise_std = noise_std
        self._dim = 388
        self._lb_vec = lb.astype(np.float32)
        self._ub_vec = ub.astype(np.float32)
        self.value = np.inf
        self.best_config = None
        self.noisy = noise_std > 0
        if self.noisy:
            warning("Using a noisy version of SVMBenchmark where training happens on a random subset of 250 points."
                    "However, the exact value of noise_std is ignored.")
        # super(SVMBenchmark, self).__init__(
        #     388, lb=np.zeros(388), ub=np.ones(388), noise_std=noise_std
        # )
        self.X, self.y = self._load_data(data_folder)
        if not self.noisy:
            np.random.seed(388)
            idxs = np.random.choice(np.arange(len(self.X)), min(10000, len(self.X)), replace=False)
            half = len(idxs) // 2
            self._X_train = self.X[idxs[:half]]
            self._X_test = self.X[idxs[half:]]
            self._y_train = self.y[idxs[:half]]
            self._y_test = self.y[idxs[half:]]
        
        # custom param
        self.input_dim = self._dim
        self.name = 'svm'
        self.bounds = [(0., 1.)]*self._dim

    def _load_data(self, data_folder: Optional[str] = None):
        if data_folder is None:
            data_folder = os.path.join(os.getcwd(), "test_functions/function_realworld_bo/svm")
        if not os.path.exists(os.path.join(data_folder, "CT_slice_X.npy")):
            sld_dir = os.path.join(data_folder, "slice_localization_data.csv.xz")
            sld_bn = os.path.basename(sld_dir)
            info(f"Slice localization data not locally available. Downloading '{sld_bn}'...")
            urllib.request.urlretrieve(
                f"http://mopta-executables.s3-website.eu-north-1.amazonaws.com/{sld_bn}",
                sld_dir)
            data = pd.read_csv(
                os.path.join(data_folder, "slice_localization_data.csv.xz")
            ).to_numpy()
            X = data[:, :385]
            y = data[:, -1]
            np.save(os.path.join(data_folder, "CT_slice_X.npy"), X)
            np.save(os.path.join(data_folder, "CT_slice_y.npy"), y)
        X = np.load(os.path.join(data_folder, "CT_slice_X.npy"))
        y = np.load(os.path.join(data_folder, "CT_slice_y.npy"))
        X = MinMaxScaler().fit_transform(X)
        y = MinMaxScaler().fit_transform(y.reshape(-1, 1)).squeeze()
        return X, y

    def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]]):
        # super(SVMBenchmark, self).__call__(x)
        x = np.array(x)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        x = x ** 2

        errors = []
        for y in x:
            C = 0.01 * (500 ** y[387])
            gamma = 0.1 * (30 ** y[386])
            epsilon = 0.01 * (100 ** y[385])
            length_scales = np.exp(4 * y[:385] - 2)

            svr = SVR(gamma=gamma, epsilon=epsilon, C=C, cache_size=1500, tol=0.001)
            if self.noisy:
                np.random.seed(None)
                idxs = np.random.choice(np.arange(len(self.X)), min(500, len(self.X)), replace=False)
                half = len(idxs) // 2
                X_train = self.X[idxs[:half]]
                X_test = self.X[idxs[half:]]
                y_train = self.y[idxs[:half]]
                y_test = self.y[idxs[half:]]
                svr.fit(X_train / length_scales, y_train)
                pred = svr.predict(X_test / length_scales)
                error = np.sqrt(np.mean(np.square(pred - y_test)))
            else:
                svr.fit(self._X_train / length_scales, self._y_train)
                pred = svr.predict(self._X_test / length_scales)
                error = np.sqrt(np.mean(np.square(pred - self._y_test)))

            errors.append(error)
            if errors[-1] < self.value:
                self.best_config = np.log(y)
                self.value = errors[-1]
        return np.array(errors).squeeze()

    def func(self, x):
        return self.__call__(x)