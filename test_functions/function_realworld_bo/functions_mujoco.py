import os
from typing import ClassVar, Dict, Optional, Tuple

import gym
import numpy as np

# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):
    def __init__(self, shape=None):
        self._n = 0
        self._M = np.zeros(shape, dtype=np.float64)
        self._S = np.zeros(shape,  dtype=np.float64)

    def copy(self):
        other = RunningStat()
        other._n = self._n
        other._M = np.copy(self._M)
        other._S = np.copy(self._S)
        return other

    def push(self, x):
        x = np.asarray(x)
        # Unvectorized update of the running statistics.
        assert x.shape == self._M.shape, ("x.shape = {}, self.shape = {}".format(x.shape, self._M.shape))
        n1 = self._n
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            delta = x - self._M
            self._M[...] += delta / self._n
            self._S[...] += delta * delta * n1 / self._n

    def update(self, other):
        n1 = self._n
        n2 = other._n
        n = n1 + n2
        delta = self._M - other._M
        delta2 = delta * delta
        M = (n1 * self._M + n2 * other._M) / n
        S = self._S + other._S + delta2 * n1 * n2 / n
        self._n = n
        self._M = M
        self._S = S

    def __repr__(self):
        return '(n={}, mean_mean={}, mean_std={})'.format(
            self.n, np.mean(self.mean), np.mean(self.std))

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

class MujucoPolicyFunc():
    ANT_ENV: ClassVar[Tuple[str, float, float, int]] = ('Ant-v4', -1.0, 1.0, 3)
    SWIMMER_ENV: ClassVar[Tuple[str, float, float, int]] = ('Swimmer-v4', -1.0, 1.0, 3)
    HALF_CHEETAH_ENV: ClassVar[Tuple[str, float, float, int]] = ('HalfCheetah-v4', -1.0, 1.0, 3)
    HOPPER_ENV: ClassVar[Tuple[str, float, float, int]] = ('Hopper-v4', -1.0, 1.0, 3)
    WALKER_2D_ENV: ClassVar[Tuple[str, float, float, int]] = ('Walker2d-v4', -1.0, 1.0, 3)
    HUMANOID_ENV: ClassVar[Tuple[str, float, float, int]] = ('Humanoid-v4', -1.0, 1.0, 3)

    def __init__(self, env: str, lb: float, ub: float, num_rollouts):
        self._env_name = env
        self._env = gym.make(env)
        self._env.reset(seed=2023)
        state_dims = self._env.observation_space.shape[0]
        action_dims = self._env.action_space.shape[0]
        self._dims = state_dims * action_dims
        self._policy_shape = (action_dims, state_dims)
        self._lb = np.full(self._dims, lb)
        self._ub = np.full(self._dims, ub)
        self._num_rollouts = num_rollouts
        self._render = False
        self._rs = RunningStat(state_dims)

        #custom parameter
        self.bounds = [(lb, ub)]*self._dims
        self.input_dim = self._dims

    # @property
    # def lb(self) -> np.ndarray:
    #     return self._lb

    # @property
    # def ub(self) -> np.ndarray:
    #     return self._ub

    @property
    def dims(self) -> int:
        return self._dims

    # @property
    # def is_minimizing(self) -> bool:
    #     return False

    def __call__(self, x):
        assert x.ndim == 1
        assert len(x) == self.dims
        assert np.all(x <= self._ub) and np.all(x >= self._lb)
        M = x.reshape(self._policy_shape)
        total_r = 0
        for _ in range(self._num_rollouts):
            obs, _ = self._env.reset()
            while True:
                self._rs.push(obs)
                norm_obs = (obs - self._rs.mean) / (self._rs.std + 1e-6)
                action = np.dot(M, norm_obs)
                obs, r, done, truncated, _ = self._env.step(action)
                total_r += r
                if done or truncated:
                    break
    
        # for minimization optimizer
        return -total_r / self._num_rollouts

    # def __str__(self):
    #     return f"Mujuco_{self._env_name}[{self.dims}]"
    
    def func(self, x: np.ndarray):
        return self.__call__(x)


func_dir = os.path.dirname(os.path.abspath(__file__))

class Humanoid(MujucoPolicyFunc):
    def __init__(self):
        super().__init__(*MujucoPolicyFunc.HUMANOID_ENV)
        self.name = 'humanoid'

class HalfCheetah(MujucoPolicyFunc):
    def __init__(self):
        super().__init__(*MujucoPolicyFunc.HALF_CHEETAH_ENV)
        self.name = 'half-cheetah'

class Hopper(MujucoPolicyFunc):
    def __init__(self):
        super().__init__(*MujucoPolicyFunc.HOPPER_ENV)
        self.name = 'hopper'
    
class Walker2d(MujucoPolicyFunc):
    def __init__(self):
        super().__init__(*MujucoPolicyFunc.WALKER_2D_ENV)
        self.name = 'walker2d'

class Swimmer(MujucoPolicyFunc):
    def __init__(self):
        super().__init__(*MujucoPolicyFunc.SWIMMER_ENV)
        self.name = 'swimmer'

class Ant(MujucoPolicyFunc):
    def __init__(self):
        super().__init__(*MujucoPolicyFunc.ANT_ENV)
        self.name = 'ant'
