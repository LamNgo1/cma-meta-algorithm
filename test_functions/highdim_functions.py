import numpy as np
from .functions_bo import hartman_6d, branin, schaffer_n2, bohachevsky_n1


class Hartmann500D:
    '''
    Hartman500 function
    '''
    def __init__(self, bounds=None):
        self.input_dim = 500

        if bounds is None:
            self.bounds = [(0.0, 1.0)]*self.input_dim
        else:
            self.bounds = bounds

        # self.min = [(0.)*self.input_dim]
        self.fmin = -3.32237
        self.name = 'hartmann500'
        self._b = hartman_6d()
        self.effective_dims = 6

    def func(self, x):
        if x.ndim == 1:
            x_ = x.reshape(1, -1)
        else:
            x_ = np.copy(x)
        return self._b.func(x_[:, :self.effective_dims])

class Branin500D:
    '''
    Branin500D function
    '''
    def __init__(self):
        self.input_dim = 500
        self._b = branin()
        self.bounds = [(0., 1.)]*self.input_dim
        # self.min = [(0.)*self.input_dim]
        self.fmin = 0.397887
        self.name = 'branin500'
        
        self.effective_dims = self._b.input_dim

    def func(self, x):
        if x.ndim == 1:
            x_ = np.copy(x.reshape(1, -1))
        else:
            x_ = np.copy(x)
        x_[:, 0] = 15.0*x_[:, 0] - 5.0
        x_[:, 1] = 15.0*x_[:, 1]
        return self._b.func(x_[:, :self.effective_dims])
    

class Branin20D:
    '''
    Branin20D function
    '''
    def __init__(self):
        self.input_dim = 20
        self._b = branin()
        self.bounds = [(0., 1.)]*self.input_dim
        # self.min = [(0.)*self.input_dim]
        self.fmin = 0.397887
        self.name = 'branin20'
        
        self.effective_dims = self._b.input_dim

    def func(self, x):
        if x.ndim == 1:
            x_ = np.copy(x.reshape(1, -1))
        else:
            x_ = np.copy(x)
        x_[:, 0] = 15.0*x_[:, 0] - 5.0
        x_[:, 1] = 15.0*x_[:, 1]
        return self._b.func(x_[:, :self.effective_dims])
    

class Branin40D:
    '''
    Branin40D function
    '''
    def __init__(self):
        self.input_dim = 40
        self._b = branin()
        self.bounds = [(0., 1.)]*self.input_dim
        # self.min = [(0.)*self.input_dim]
        self.fmin = 0.397887
        self.name = 'branin40'
        
        self.effective_dims = self._b.input_dim

    def func(self, x):
        if x.ndim == 1:
            x_ = np.copy(x.reshape(1, -1))
        else:
            x_ = np.copy(x)
        x_[:, 0] = 15.0*x_[:, 0] - 5.0
        x_[:, 1] = 15.0*x_[:, 1]
        return self._b.func(x_[:, :self.effective_dims])

class Schaffer40:
    '''
    Schaffer-N2 40D function
    '''
    def __init__(self, bounds=None):
        self.input_dim = 40

        if bounds is None:
            self.bounds = [(-100., 100.)]*self.input_dim
        else:
            self.bounds = bounds

        # self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.name = 'schaffer40'
        self._b = schaffer_n2()
        self.effective_dims = 2

    def func(self, x):
        if x.ndim == 1:
            x_ = np.copy(x.reshape(1, -1))
        else:
            x_ = np.copy(x)
        return self._b.func(x_[:, :self.effective_dims])

class Schaffer100:
    '''
    Schaffer-N2 100D function
    '''
    def __init__(self, bounds=None):
        self.input_dim = 100

        if bounds is None:
            self.bounds = [(-100., 100.)]*self.input_dim
        else:
            self.bounds = bounds

        # self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.name = 'schaffer100'
        self._b = schaffer_n2()
        self.effective_dims = 2

    def func(self, x):
        if x.ndim == 1:
            x_ = np.copy(x.reshape(1, -1))
        else:
            x_ = np.copy(x)
        return self._b.func(x_[:, :self.effective_dims])
    

class Bohachevsky100:
    '''
    bohachevsky-N1 100D function
    '''
    def __init__(self, bounds=None):
        self.input_dim = 100

        if bounds is None:
            self.bounds = [(-100., 100.)]*self.input_dim
        else:
            self.bounds = bounds

        # self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.name = 'bohachevsky100'
        self._b = bohachevsky_n1()
        self.effective_dims = 2

    def func(self, x):
        if x.ndim == 1:
            x_ = np.copy(x.reshape(1, -1))
        else:
            x_ = np.copy(x)
        return self._b.func(x_[:, :self.effective_dims])