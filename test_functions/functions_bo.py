# -*- coding: utf-8 -*-
"""
@author:
@description: Some common synthetic functions for testing Bayesian Optimization
"""


import numpy as np
from collections import OrderedDict

trajectory_plot = False

def reshape(x, input_dim):
    '''
    Reshapes x into a matrix with input_dim columns

    '''
    x = np.array(x)
    if x.size == input_dim:
        x = x.reshape((1, input_dim))
    return x


class functions:
    def plot(self):
        print("not implemented")

class hartman_6d():
    '''
    Hartman6 function
    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 6

        if bounds is None:
            self.bounds = [(0, 1)]*self.input_dim
        else:
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = -3.32237
        self.ismax = 1
        self.name = 'hartman_6d'

    def func(self, X):
        X = reshape(X, self.input_dim)
        n = X.shape[0]

        alpha = [1.0, 1.2, 3.0, 3.2]

        A = [[10, 3, 17, 3.5, 1.7, 8],
             [0.05, 10, 17, 0.1, 8, 14],
             [3, 3.5, 1.7, 10, 17, 8],
             [17, 8, 0.05, 10, 0.1, 14]]
        A = np.asarray(A)
        P = [[1312, 1696, 5569, 124, 8283, 5886],
             [2329, 4135, 8307, 3736, 1004, 9991],
             [2348, 1451, 3522, 2883, 3047, 6650],
             [4047, 8828, 8732, 5743, 1091, 381]]

        P = np.asarray(P)
        c = 10**(-4)
        P = np.multiply(P, c)
        outer = 0

        fval = np.zeros((n, 1))
        for idx in range(n):
            outer = 0
            for ii in range(4):
                inner = 0
                for jj in range(6):
                    xj = X[idx, jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                    inner = inner + Aij*(xj-Pij)**2
                new = alpha[ii] * np.exp(-inner)
                outer = outer + new
            fval[idx] = -(2.58 + outer) / 1.94

        if (n == 1):
            return self.ismax*(fval[0][0])
        else:
            return self.ismax*(fval)


class branin(functions):
    def __init__(self):
        self.input_dim = 2
        self.bounds = [(-5, 10), (0, 15)]
        self.fmin = 0.397887
        self.min = [[-3.1415, 12.275],[3.1415, 2.275],[9.42478, 2.475]]
        self.ismax = 1
        self.name = 'branin'

    def func(self, X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            x1 = X[0]
            x2 = X[1]
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
        a = 1
        b = 5.1/(4*np.pi*np.pi)
        c = 5/np.pi
        r = 6
        s = 10
        t = 1/(8*np.pi)
        fx = a*(x2-b*x1*x1+c*x1-r)**2+s*(1-t)*np.cos(x1)+s
        return fx*self.ismax

class branin_uniformbound(functions):
    def __init__(self):
        self.input_dim = 2
        self.bounds = [(0, 1)]*2
        self.fmin = 0.397887
        xmins = [[-3.1415, 12.275],[3.1415, 2.275],[9.42478, 2.475]]
        self.min = []
        for xmin in xmins:
            xmin[0] = (xmin[0]+5.0)/15.0
            xmin[1] = xmin[1]/15.0
            self.min.append(xmin)
        
        self.ismax = 1
        self.name = 'branin'

    def func(self, X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            x1 = X[0]
            x2 = X[1]
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]

        x1 = 15.0*x1 - 5.0
        x2 = 15.0*x2

        a = 1
        b = 5.1/(4*np.pi*np.pi)
        c = 5/np.pi
        r = 6
        s = 10
        t = 1/(8*np.pi)
        fx = a*(x2-b*x1*x1+c*x1-r)**2+s*(1-t)*np.cos(x1)+s
        return fx*self.ismax
    
class gSobol:
    '''
    gSolbol function

    param a: one-dimensional array containing the coefficients of the function
    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, input_dim, bounds=None, sd=None):
        self.input_dim = input_dim

        a = np.zeros((1, self.input_dim))
        for i in range(1, self.input_dim+1):
            if (i == 1) | (i == 2):
                a[0, i-1] = 0
            else:
                a[0, i-1] = 6.52
        self.a = a

        if bounds is None:
            self.bounds = [(0, 1)]*self.input_dim
        else:
            self.bounds = bounds

        self.S_coef = (1/(3*((1+self.a)**2))) / (np.prod(1+1/(3*((1+self.a)**2)))-1)
        if sd is None:
            self.sd = 0
        else:
            self.sd = sd

        self.ismax = -1
        self.fmin = 0
        self.name = 'gSobol'

    def func(self, X):
        X = reshape(X, self.input_dim)
        n = X.shape[0]
        aux = (abs(4*X-2)+np.ones(n).reshape(n, 1)*self.a) / (1+np.ones(n).reshape(n, 1)*self.a)
        fval = np.cumprod(aux, axis=1)[:, self.input_dim-1]

        return self.ismax*fval


class hartman_4d:
    '''
    hartman_4d function
    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 4

        if bounds is None:
            self.bounds = [(0, 1)]*self.input_dim
        else:
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = -3.32237
        self.ismax = -1
        self.name = 'hartman_4d'

    def func(self, X):
        X = reshape(X, self.input_dim)
        n = X.shape[0]

        alpha = [1.0, 1.2, 3.0, 3.2]

        A = [[10, 3, 17, 3.5, 1.7, 8],
             [0.05, 10, 17, 0.1, 8, 14],
             [3, 3.5, 1.7, 10, 17, 8],
             [17, 8, 0.05, 10, 0.1, 14]]
        A = np.asarray(A)
        P = [[1312, 1696, 5569, 124, 8283, 5886],
             [2329, 4135, 8307, 3736, 1004, 9991],
             [2348, 1451, 3522, 2883, 3047, 6650],
             [4047, 8828, 8732, 5743, 1091, 381]]

        P = np.asarray(P)
        c = 10**(-4)
        P = np.multiply(P, c)
        outer = 0

        fval = np.zeros((n, 1))
        for idx in range(n):
            X_idx = X[idx, :]
            outer = 0
            for ii in range(4):
                inner = 0
                for jj in range(4):
                    xj = X_idx[jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                inner = inner + Aij*(xj-Pij)**2
                new = alpha[ii] * np.exp(-inner)
                outer = outer + new
            fval[idx] = (1.1 - outer) / 0.839
        if n == 1:
            return self.ismax*(fval[0][0])
        else:
            return self.ismax*(fval)


class hartman_3d:
    '''
    hartman_3d: function
    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 3

        if bounds is None:
            self.bounds = [(0, 1)]*self.input_dim
        else:
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = -3.86278
        self.ismax = -1
        self.name = 'hartman_3d'

    def func(self, X):
        X = reshape(X, self.input_dim)
        n = X.shape[0]

        alpha = [1.0, 1.2, 3.0, 3.2]
        A = [[3.0, 10, 30],
             [0.1, 10, 35],
             [3.0, 10, 30],
             [0.1, 10, 35]]
        A = np.asarray(A)
        P = [[3689, 1170, 2673],
             [4699, 4387, 7470],
             [1091, 8732, 5547],
             [381, 5743, 8828]]

        P = np.asarray(P)
        c = 10**(-4)
        P = np.multiply(P, c)
        outer = 0

        fval = np.zeros((n, 1))
        for idx in range(n):
            outer = 0
            for ii in range(4):
                inner = 0
                for jj in range(3):
                    xj = X[idx, jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                    inner = inner + Aij*(xj-Pij)**2
                new = alpha[ii] * np.exp(-inner)
                outer = outer + new
            fval[idx] = -outer

        if n == 1:
            return self.ismax*(fval[0][0])
        else:
            return self.ismax*(fval)
    
class ackley:
    '''
    Ackley function 

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self, input_dim, bounds=None,sd=None):
        self.input_dim = input_dim

        if bounds == None: 
            self.bounds =[(-32.768,32.768)]*self.input_dim
        else: 
            self.bounds = bounds

        self.min = np.zeros(self.input_dim)
        self.fmin = 0
        self.ismax = 1
        self.name = 'ackley'
        
    def func(self,X):
        X = reshape(X,self.input_dim)
        fval = (20+np.exp(1)-20*np.exp(-0.2*np.sqrt((X**2).sum(-1)/self.input_dim))-np.exp(np.cos(2*np.pi*X).sum(-1)/self.input_dim))
        
      
        return self.ismax*fval
    

class shifted_ackley:
    '''
    Ackley function 

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self, input_dim, bounds=None,sd=None):
        self.input_dim = input_dim

        if bounds == None: 
            self.bounds =[(-32.768,32.768)]*self.input_dim
        else: 
            self.bounds = bounds

        self.min = np.zeros((1, self.input_dim))
        self.fmin = 0
        self.ismax = 1
        self.name = 'shifted-ackley'
        np.random.seed(2023)
        self.offset = np.random.uniform(-32.768, 32.768, input_dim).reshape(1, -1)
        print(self.offset)
        self.min -= self.offset
        
    def func(self,X):
        X = reshape(X,self.input_dim) + self.offset
        fval = (20+np.exp(1)-20*np.exp(-0.2*np.sqrt((X**2).sum(-1)/self.input_dim))-np.exp(np.cos(2*np.pi*X).sum(-1)/self.input_dim))
        
      
        return self.ismax*fval


class beale(functions):
    '''
    beale function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-4.5, 4.5), (-4.5, 4.5)]
        else:
            self.bounds = bounds
        self.min = np.array([3, 0.5])
        self.fmin = 0
        self.ismax = 1
        self.name = 'Beale'

    def func(self, X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            x1 = X[0]
            x2 = X[1]
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
        fval = (1.5-x1+x1*x2)**2+(2.25-x1+x1*x2**2)**2+(2.625-x1+x1*x2**3)**2
        return self.ismax*fval


class egg_holder(functions):
    '''
    Egg holder function
    '''
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-512, 512), (-512, 512)]
        else:
            self.bounds = bounds
        self.min = np.array([512, 404.2319])
        self.fmin = 0
        self.ismax = 1
        self.name = 'Eggholder'

    def func(self, X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            x1 = X[0]
            x2 = X[1]
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
        fval = -(x2+47)*np.sin(np.sqrt(np.abs(x2+x1/2+47))) - x1*np.sin(np.sqrt(np.abs(x1-x2-47)))
        return self.ismax*fval


class Levy(functions):
    '''
    Levy function
    '''
    def __init__(self, input_dim, bounds=None, sd=None):
        self.input_dim = input_dim
        if bounds is None:
            self.bounds = [(-10.0, 10.0)]*self.input_dim
        else:
            self.bounds = bounds
        self.min = [(1.)]*self.input_dim
        self.fmin = 0
        self.ismax = 1
        self.name = 'levy'

    def func(self, X):
        X = reshape(X, self.input_dim)

        w = np.zeros((X.shape[0], self.input_dim))
        for i in range(1, self.input_dim+1):
            w[:, i-1] = 1 + 1/4*(X[:, i-1]-1)

        fval = (np.sin(np.pi*w[:, 0]))**2 + ((w[:, self.input_dim-1]-1)**2)*(1+(np.sin(2*np.pi*w[:, self.input_dim-1]))**2)
        for i in range(1, self.input_dim):
            fval += ((w[:, i]-1)**2)*(1+10*(np.sin(np.pi*w[:, i]))**2) 

        return self.ismax*fval
    

class shifted_levy(functions):
    '''
    Shifted Levy function
    '''
    def __init__(self, input_dim, bounds=None, sd=None):
        self.input_dim = input_dim
        if bounds is None:
            self.bounds = [(-10.0, 10.0)]*self.input_dim
        else:
            self.bounds = bounds
        self.min = [(1.)]*self.input_dim
        self.fmin = 0
        self.ismax = 1
        self.name = 'shifted-levy'
        np.random.seed(2024)
        if input_dim == 2 and trajectory_plot:
            self.offset = np.array([-3,4])
        else:
            self.offset = np.random.uniform(-10.0, 10.0, input_dim).reshape(1, -1)
        print(self.offset)
        self.min -= self.offset

    def func(self, X):
        X = reshape(X, self.input_dim) + self.offset

        w = np.zeros((X.shape[0], self.input_dim))
        for i in range(1, self.input_dim+1):
            w[:, i-1] = 1 + 1/4*(X[:, i-1]-1)

        fval = (np.sin(np.pi*w[:, 0]))**2 + ((w[:, self.input_dim-1]-1)**2)*(1+(np.sin(2*np.pi*w[:, self.input_dim-1]))**2)
        for i in range(1, self.input_dim):
            fval += ((w[:, i]-1)**2)*(1+10*(np.sin(np.pi*w[:, i]))**2) 

        return self.ismax*fval


class rosenbrock(functions):
    '''
    rosenbrock function
    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, input_dim, bounds=None):
        self.input_dim = input_dim
        if bounds is None:
            self.bounds = [(-2.048, 2.048)]*self.input_dim
        else:
            self.bounds = bounds

        self.min = [(0.)]*self.input_dim
        self.fmin = 0
        self.ismax = 1
        self.name = 'rosenbrock'
    
    def func(self, X):
        X = reshape(X, self.input_dim)
        fval = 0
        for i in range(self.input_dim-1):
            fval += (100*(X[:, i+1]-X[:, i]**2)**2 + (X[:, i]-1)**2)
        
        return self.ismax*fval


class alpine:
    '''
    alpine function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, input_dim, bounds=None, sd=None):
        if bounds is None:
            self.bounds = [(-10.0, 10.0)]*input_dim
        else:
            self.bounds = bounds
        self.min = [(0)]*input_dim
        self.fmin = 0
        self.input_dim = input_dim
        if sd is None:
            self.sd = 0
        else:
            self.sd = sd

        self.ismax = 1
        self.name = 'alpine'

    def func(self, X):
        X = reshape(X, self.input_dim)
        temp = abs(X*np.sin(X) + 0.1*X)
        if len(temp.shape) <= 1:
            fval = np.sum(temp)
        else:
            fval = np.sum(temp, axis=1)

        return self.ismax*fval

class shifted_alpine:
    '''
    shifted alpine function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, input_dim, bounds=None, sd=None):
        if bounds is None:
            self.bounds = [(-10.0, 10.0)]*input_dim
        else:
            self.bounds = bounds
        self.min = [(0)]*input_dim
        self.fmin = 0
        self.input_dim = input_dim
        if sd is None:
            self.sd = 0
        else:
            self.sd = sd

        self.ismax = 1
        self.name = 'shifted-alpine'
        np.random.seed(2025)
        if input_dim == 2 and trajectory_plot:
            self.offset = np.array([4,5])
        else:
            self.offset = np.random.uniform(-10.0, 10.0, input_dim).reshape(1, -1)
        print(self.offset)
        self.min -= self.offset

    def func(self, X):
        X = reshape(X, self.input_dim) + self.offset
        temp = abs(X*np.sin(X) + 0.1*X)
        if len(temp.shape) <= 1:
            fval = np.sum(temp)
        else:
            fval = np.sum(temp, axis=1)

        return self.ismax*fval


class sixhumpcamel(functions):
    '''
    Six hump camel function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-2, 2)]*self.input_dim
        else:
            self.bounds = bounds
        self.min = [(0.0898, -0.7126), (-0.0898, 0.7126)]
        self.fmin = -1.0316
        self.ismax = -1

        self.name = 'Six-hump camel'

    def func(self, X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            x1 = X[0]
            x2 = X[1]
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
        term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2
        term2 = x1*x2
        term3 = (-4+4*x2**2) * x2**2
        fval = term1 + term2 + term3
        return self.ismax*fval


class schaffer_n2(functions):
    def __init__(self):
        self.input_dim = 2
        self.bounds = [(-100, 100), (-100, 100)]
        self.fmin = 0
        self.min = [0, 0]
        self.ismax = 1
        self.name = 'schaffer-n2'

    def func(self, X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            x1 = X[0]
            x2 = X[1]
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
        
        fx = 0.5 + (np.sin(x1**2 - x2**2)**2 - 0.5)/(1+0.001*(x1**2+x2**2))**2

        return fx*self.ismax
    

class bohachevsky_n1(functions):
    def __init__(self):
        self.input_dim = 2
        self.bounds = [(-100, 100), (-100, 100)]
        self.fmin = 0
        self.min = [0, 0]
        self.ismax = 1
        self.name = 'bohachevsky-n1'

    def func(self, X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            x1 = X[0]
            x2 = X[1]
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
        
        fx = x1**2 + 2*x2**2 - 0.3* np.cos(3*np.pi*x1) - 0.4*np.cos(4*np.pi*x2) + 0.7

        return fx*self.ismax


class damavandi(functions):
    def __init__(self):
        self.input_dim = 2
        self.bounds = [(0, 14), (0, 14)]
        self.fmin = 0
        self.min = [2, 2]
        self.ismax = -1
        self.name = 'damavandi'

    def func(self, X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            x1 = X[0]
            x2 = X[1]
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
        
        fx = (1-np.abs(np.sin(np.pi*(x1-2))*np.sin(np.pi*(x2-2))/(np.pi**2*(x1-2)*(x2-2)))**5)*(2+(x1-7)**2+2*(x2-7)**2)

        return fx*self.ismax
      

class h1(functions):
    '''
    h1 function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None, sd=None):
        self.input_dim = 2
        if bounds is None:
            self.bounds = [(-100, 100), (-100, 100)]
        else:
            self.bounds = bounds
        self.min = [(8.6998, 6.7665)]
        self.fmin = 2
        self.ismax = 1
        self.name = 'Beale'

    def func(self, X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            x1 = X[0]
            x2 = X[1]
        else:
            x1 = X[:, 0]
            x2 = X[:, 1]
        fval = (np.sin((x1-x2/8))**2 + np.sin((x2+x1/8))**2)/np.sqrt((x1-8.6998)**2+(x2-6.7665)**2+1)
        return self.ismax*fval


class Shekel(functions):
    '''
    Shekel function
    '''
    def __init__(self, input_dim, bounds=None, sd=None):
        if input_dim > 4:
            raise AssertionError('Input dim must be smaller/equal to 4!')
      
        self.input_dim = input_dim
        if bounds is None:
            self.bounds = [(0, 10)]*self.input_dim
        else:
            self.bounds = bounds
        self.min = -10.5364
        self.fmin = 0
        self.ismax = -1
        self.name = 'Shekel'

    def func(self, X):
        m = 10
        beta = 1/m*np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
        C = np.array([[4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                      [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
                      [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                      [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6]])

        X = np.asarray(X)
        if len(X.shape) == 1:
            if self.input_dim == 1:
                x1 = X[0]
            if self.input_dim == 2:
                x1 = X[0]
                x2 = X[1]
            if self.input_dim == 3:
                x1 = X[0]
                x2 = X[1]
                x3 = X[2]
            if self.input_dim == 4:
                x1 = X[0]
                x2 = X[1]
                x3 = X[2]
                x4 = X[3]
        else:
            if self.input_dim == 1:
                x1 = X[:, 0]
            if self.input_dim == 2:
                x1 = X[:, 0]
                x2 = X[:, 1]
            if self.input_dim == 3:
                x1 = X[:, 0]
                x2 = X[:, 1]
                x3 = X[:, 2]
            if self.input_dim == 4:
                x1 = X[:, 0]
                x2 = X[:, 1]
                x3 = X[:, 2]
                x4 = X[:, 3]

        fval = 0
        for i in range(m):
            if self.input_dim == 1:
                fval += -np.divide(1, beta[i] + (x1-C[0, i])**2)
            if self.input_dim == 2:
                fval += -np.divide(1, beta[i] + (x1-C[0, i])**2 + (x2-C[1, i])**2)
            if self.input_dim == 3:
                fval += -np.divide(1, beta[i] + (x1-C[0, i])**2 + (x2-C[1, i])**2 + (x3-C[2, i])**2)
            if self.input_dim == 4:
                fval += -np.divide(1, beta[i] + (x1-C[0, i])**2 + (x2-C[1, i])**2 + (x3-C[2, i])**2 + (x4-C[3, i])**2)

        return self.ismax*fval


class schwefel(functions):
    '''
    schwefel function
    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, input_dim, bounds=None):
        self.input_dim = input_dim
        if bounds is None:
            self.bounds = [(-500, 500)]*self.input_dim
        else:
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax = 1
        self.name = 'schwefel'
    
    def func(self, X):
        X = reshape(X, self.input_dim)
        fval = 418.9829*self.input_dim
        for i in range(self.input_dim):
            fval -= (X[:, i]*np.sin(np.sqrt(np.abs(X[:, i]))))
        
        return self.ismax*fval


class rastrigin(functions):
    '''
    rastrigin function
    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, input_dim, bounds=None):
        self.input_dim = input_dim
        if bounds is None:
            self.bounds = [(-5.12, 5.12)]*self.input_dim
        else:
            self.bounds = bounds

        self.min = np.zeros(self.input_dim) 
        self.fmin = 0
        self.ismax = 1
        self.name = 'rastrigin'

    def func(self, X):
        X = reshape(X, self.input_dim) 
        fval = 0
        for i in range(self.input_dim):
            fval += (X[:, i]**2 - 10*np.cos(2*np.pi*X[:, i]))
        fval += 10*self.input_dim
        
        return self.ismax*fval 


class dropwave(functions):
    '''
    Dropwave function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(-1,1),(-1,1)]
        else: self.bounds = bounds
        self.min = [(0, 0)]
        self.fmin = 0
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'dropwave'

    def func(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            fval = - (1+np.cos(12*np.sqrt(X[:,0]**2+X[:,1]**2))) / (0.5*(X[:,0]**2+X[:,1]**2)+2) 
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return -fval.reshape(n,1) + noise


class goldstein(functions):
    '''
    Goldstein function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds is  None: self.bounds = [(-2,2),(-2,2)]
        else: self.bounds = bounds
        self.min = [(0,-1)]
        self.fmin = 3
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Goldstein'

    def func(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        if X.shape[1] != self.input_dim:
            return 'Wrong input dimension'
        else:
            x1 = X[:,0]
            x2 = X[:,1]
            fact1a = (x1 + x2 + 1)**2
            fact1b = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
            fact1 = 1 + fact1a*fact1b
            fact2a = (2*x1 - 3*x2)**2
            fact2b = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
            fact2 = 30 + fact2a*fact2b
            fval = fact1*fact2
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return -fval.reshape(n,1) + noise


class Benchmark(object):
    """
    Defines a global optimization benchmark problem.
    
    This abstract class defines the basic structure of a global
    optimization problem. Subclasses should implement the ``evaluator`` method
    for a particular optimization problem.
        
    Public Attributes:
    
    - *dimensions* -- the number of inputs to the problem
    - *fun_evals* -- stores the number of function evaluations, as some crappy
      optimization frameworks (i.e., `nlopt`) do not return this value
    - *change_dimensionality* -- whether we can change the benchmark function `x`
      variable length (i.e., the dimensionality of the problem)
    - *custom_bounds* -- a set of lower/upper bounds for plot purposes (if needed).
    - *spacing* -- the spacing to use to generate evenly spaced samples across the
      lower/upper bounds on the variables, for plotting purposes
    """

    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.fun_evals = 0
        self.change_dimensionality = False
        self.custom_bounds = None

        if dimensions == 1:
            self.spacing = 1001
        else:
            self.spacing = 201
        
    def __str__(self):
        return '{0} ({1} dimensions)'.format(self.__class__.__name__, self.dimensions)
        
    def __repr__(self):
        return self.__class__.__name__
    
    def generator(self):
        """The generator function for the benchmark problem."""
        return [np.random.uniform(l, u) for l, u in self.bounds]
        
    def evaluator(self, candidates):
        """The evaluator function for the benchmark problem."""
        raise NotImplementedError

    def set_dimensions(self, ndim):
        self.dimensions = ndim

    def lower_bounds_constraints(self, x):

        lower  = np.asarray([b[0] for b in self.bounds])
        return np.asarray(x) - lower
    

    def upper_bounds_constraints(self, x):

        upper  = np.asarray([b[1] for b in self.bounds])
        return upper - np.asarray(x)


class Deceptive(Benchmark):
    """
    Deceptive test objective function.
    This class defines the Deceptive global optimization problem. This 
    is a multimodal minimization problem defined as follows:
    """
    
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        
        self.bounds = [(0, 1)]*dimensions

        self.input_dim = dimensions
        n = self.dimensions
        alpha = np.arange(1.0, n + 1.0)/(n + 1.0)

        self.global_optimum = alpha
        self.fglob = -1.0
        self.change_dimensionality = True
        self.ismax = -1
        
    def func(self, x, *args):

        self.fun_evals += 1

        n = self.dimensions
        alpha = np.arange(1.0, n + 1.0)/(n + 1.0)
        beta = 2.0

        g = np.zeros((n, ))
        
        x = x.ravel()
        for i in range(n):
            if x[i] <= 0.0:
                g[i] = x[i]
            elif x[i] < 0.8*alpha[i]:
                g[i] = -x[i]/alpha[i] + 0.8
            elif x[i] < alpha[i]:
                g[i] = 5.0*x[i]/alpha[i] - 4.0
            elif x[i] < (1.0 + 4*alpha[i])/5.0:
                g[i] = 5.0*(x[i] - alpha[i])/(alpha[i] - 1.0) + 1.0
            elif x[i] <= 1.0:
                g[i] = (x[i] - 1.0)/(1.0 - alpha[i]) + 4.0/5.0
            else:
                g[i] = x[i] - 1.0
        
        return (-((1.0/n)*sum(g))**beta)*self.ismax

class powell(functions):
    '''
    alpine function

    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self, input_dim, bounds=None):
        if bounds is None:
            self.bounds = [(-4, 5)]*input_dim
        else:
            self.bounds = bounds
        self.min = [(0)]*input_dim
        self.fmin = 0
        self.input_dim = input_dim

        self.ismax = 1
        self.name = 'powell'

    def func(self, X):
        assert X.ndim == 1
        X = reshape(X, self.input_dim)
        fval = self.powell(X)
        return self.ismax*fval
    
    def powell(self, x):
        x = np.asarray_chkfinite(x)
        n = self.input_dim
        n4 = ((n + 3) // 4) * 4
        if n < n4:
            x = np.append( x, np.zeros( n4 - n ))
        x = x.reshape(( 4, -1 ))  # 4 rows: x[4i-3] [4i-2] [4i-1] [4i]
        f = np.empty_like( x )
        f[0] = x[0] + 10 * x[1]
        f[1] = np.sqrt(5) * (x[2] - x[3])
        f[2] = (x[1] - 2 * x[2]) **2
        f[3] = np.sqrt(10) * (x[0] - x[3]) **2
        return np.sum( f**2 )
    
class ellipsoid:
    '''
    Axis Parallel Hyper-Ellipsoid function 

    '''
    def __init__(self, input_dim, bounds=None):
        self.input_dim = input_dim

        if bounds == None: 
            self.bounds =[(-10.,10.)]*self.input_dim
        else: 
            self.bounds = bounds

        self.min = np.zeros(self.input_dim)
        self.fmin = 0
        self.ismax = 1
        self.name = 'ellipsoid'
        
    def func(self,X):
        X = reshape(X,self.input_dim)

        fval = np.array([self.eval_one(x) for x in X])
      
        return self.ismax*fval
    
    def eval_one(self, x):
        assert x.ndim == 1
        assert len(x) == self.input_dim
        fval = np.sum(np.arange(1, len(x)+1) * np.square(x))
        return fval

class shifted_ellipsoid:
    '''
    Shifted Axis Parallel Hyper-Ellipsoid function 

    '''
    def __init__(self, input_dim, bounds=None):
        self.input_dim = input_dim

        if bounds == None: 
            self.bounds =[(-10.,10.)]*self.input_dim
        else: 
            self.bounds = bounds

        self.min = np.zeros((1, self.input_dim))
        self.fmin = 0
        self.ismax = 1
        self.name = 'shifted-ellipsoid'

        np.random.seed(2026)
        self.offset = np.random.uniform(-10., 10., input_dim).reshape(1, -1)
        print(self.offset)
        self.min -= self.offset
        
    def func(self,X):
        X = reshape(X,self.input_dim) + self.offset

        fval = np.array([self.eval_one(x) for x in X])
      
        return self.ismax*fval
    
    def eval_one(self, x):
        assert x.ndim == 1
        assert len(x) == self.input_dim
        fval = np.sum(np.arange(1, len(x)+1) * np.square(x))
        return fval