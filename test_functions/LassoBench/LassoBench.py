#!/usr/bin/python
"""
==================================================
LassoBench
High-Dimensional Hyperparameter
Optimization Benchmark
Contact: kenan.sehic@cs.lth.se
=================================================
"""
import numpy as np
from celer import Lasso, LassoCV

from sparse_ho.models import WeightedLasso
from sparse_ho.criterion import HeldOutMSE, CrossVal
from sparse_ho.utils import Monitor
from sparse_ho import Implicit, ImplicitForward, grad_search
from sparse_ho.optimizers import LineSearch, GradientDescent, Adam

from celer.datasets import make_correlated_data

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

from libsvmdata import fetch_libsvm

import timeit


class SyntheticBenchmark():
    """
    Creating a synthetic benchmark for a HPO algorithm.

    ...

    Attributes
    ----------
    pick_bench : str
        name of a predefined bench such as
        synt_simple
        synt_medium
        synt_high
        synt_hard
    noise : str, optional
        increasing the noise level for the predefined bench
    mf_opt : str, optional
        name of a multi-fidelity framework
        continuous_fidelity or discrete_fidelity
    n_features : int, optional
        number of features in design matrix i.e. the dimension of search space
    n_samples : int, optional
        number of samples in design matrix
    snr_level : int, optional
        level of noise, 1 very noisy 10 noiseless
    corr_level : int, optional
        correlation between features in design matrix
    n_nonzeros: int, optional
        number of nonzero elements in true reg coef
    tol_level: int, optional
        tolerance level for inner opt part
    w_true: array, optional
        custimized true regression coefficients
    n_splits: int, optional
        number of splits in CV
    test_size: int, optional
        percentage of test data
    seed: int, optional
        seed number

    Methods
    -------
    evaluate(input_config):
        Return cross-validation loss divided by oracle for configuration.
    fidelity_evaluate(input_config, index_fidelity=None):
        Return cross-validation loss for configuration and fidelity index.
    test(input_config):
        Return post-processing metrics MSPE divided by oracle error,
        Fscore and reg coef for configuration.
    run_LASSOCV(n_alphas=100):
        Running baseline LASSOCV and return loss, mspe divided by oracle,
        fsore, best-found 1D config and time elapsed.
    run_sparseho(grad_solver='gd', algo_pick='imp_forw', n_steps=10, init_point=None, verbose=False):
        Running basedline Sparse-HO and return loss, mspe divided by oracle,
        fscore and time elapsed.
    """

    def __init__(self, pick_bench=None, noise=False, mf_opt=None, n_features=1280, n_samples=640,
                 snr_level=1, corr_level=0.6, n_nonzeros=10, tol_level=1e-4,
                 w_true=None, n_splits=5, test_size=0.15, seed=42):
        """
        Constructs all the necessary attributes for synt bench.

        Parameters
        ----------
            pick_bench : str
                name of a predefined bench such as
                synt_simple
                synt_medium
                synt_high
                synt_hard
            noise : str, optional
                increasing the noise level for the predefined bench (default: False)
            mf_opt : str, optional
                name of a multi-fidelity framework
                continuous_fidelity or discrete_fidelity
            n_features : int, optional
                number of features in design matrix i.e. the dimension of search space
            n_samples : int, optional
                number of samples in design matrix
            snr_level : int, optional
                level of noise, 1 very noisy 10 noiseless
            corr_level : int, optional
                correlation between features in design matrix
            n_nonzeros: int, optional
                number of nonzero elements in true reg coef
            tol_level: int, optional
                tolerance level for inner opt part
            w_true: array, optional
                customized true regression coefficients
            n_splits: int, optional
                number of splits in CV
            test_size: int, optional
                percentage of test data
            seed: int, optional
                seed number
        """

        if pick_bench is not None:

            if noise is True:
                snr_level = 3
            else:
                snr_level = 10

            if pick_bench.lower() == 'synt_simple':
                n_features = 60
                n_samples = 30
                corr_level = 0.6
                w_true = np.zeros(n_features)
                size_supp = 3
                w_true[::n_features // size_supp] = (-1) ** np.arange(size_supp)
            elif pick_bench.lower() == 'synt_medium':
                n_features = 100
                n_samples = 50
                corr_level = 0.6
                w_true = np.zeros(n_features)
                size_supp = 5
                w_true[::n_features // size_supp] = (-1) ** np.arange(size_supp)
            elif pick_bench.lower() == 'synt_high':
                n_features = 300
                n_samples = 150
                corr_level = 0.6
                w_true = np.zeros(n_features)
                size_supp = 15
                w_true[::n_features // size_supp] = (-1) ** np.arange(size_supp)
            elif pick_bench.lower() == 'synt_hard':
                n_features = 1000
                n_samples = 500
                corr_level = 0.6
                w_true = np.zeros(n_features)
                size_supp = 50
                w_true[::n_features // size_supp] = (-1) ** np.arange(size_supp)
            else:
                raise ValueError(
                    "Please select one of the predefined benchmarks or creat your own.")

        self.mf = 2

        if mf_opt is not None:
            if mf_opt == 'continuous_fidelity':
                self.mf = 0
            elif mf_opt == 'discrete_fidelity':
                self.mf = 1
            else:
                raise ValueError(
                    "Please select one of two mf options continuous_fidelity or discrete_fidelity.")

        self.tol_level = tol_level
        self.n_features = n_features
        self.n_splits = n_splits

        X, y, self.w_true = make_correlated_data(
            n_samples=n_samples, n_features=n_features,
            corr=corr_level, w_true=w_true,
            snr=snr_level, density=n_nonzeros/n_features,
            random_state=seed)

        # split train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed)

        self.kf = KFold(shuffle=True, n_splits=self.n_splits, random_state=seed)

        self.alpha_max = np.max(np.abs(
            self.X_train.T @ self.y_train)) / len(self.y_train)
        self.alpha_min = self.alpha_max / 1e2

        self.log_alpha_min = np.log(self.alpha_min)
        self.log_alpha_max = np.log(self.alpha_max)

        self.eps_support = 1e-6

        self.coef_true_support = np.abs(self.w_true) > self.eps_support
        self.mspe_oracle = mean_squared_error(
            self.X_test @ self.w_true, self.y_test)
        self.loss_oracle = mean_squared_error(
            self.X_train @ self.w_true, self.y_train)

    def scale_domain(self, x):
        """Scaling the input configuration to the original Lasso space

        Args:
            x (numpy array): the input configuration scaled to [-1, 1]

        Returns:
           x_copy (numpy array): the original Lasso input configuration
        """
        x_copy = np.copy(x)
        x_copy = x_copy * (
                self.log_alpha_max - self.log_alpha_min) / 2 + (
                    self.log_alpha_max + self.log_alpha_min) / 2
        return x_copy

    def evaluate(self, input_config):
        """
        Evaluate configuration for synt bench

        Parameters
        ----------
        input_config : array size of n_features

        Returns
        -------
        Cross-validation Loss divided by oracle. The goal is to be close or less than 1.
        """
        if np.any(input_config < -1) or np.any(input_config > 1):
            raise ValueError(
                "The configuration is outside the bounds.")

        scaled_x = self.scale_domain(input_config)

        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=False)
        model = WeightedLasso(estimator=estimator)
        monitor = Monitor()
        sub_criterion = HeldOutMSE(None, None)
        criterion = CrossVal(sub_criterion, cv=self.kf)
        val_loss = criterion.get_val(model, self.X_train, self.y_train,
                                     log_alpha=scaled_x,
                                     monitor=monitor, tol=self.tol_level)

        return val_loss/self.loss_oracle

    def fidelity_evaluate(self, input_config, index_fidelity=None):
        """
        Return cross-validation loss for selected fidelity.

        Parameters
        ----------
        input_config : array size of n_features
        index_fidelity : int, optional
            If mf_opt is selected, then selecting which fidelity to evaluate. (default is None)
            For continuous_fidelity, index_fidelity is defined within [0,1]
            For discrete_fidelity, index_fidelity is a dicreate par between 0 and 5.

        Returns
        -------
        Cross-validation Loss divided by oracle. The goal is to be close or less than 1.
        """
        if self.mf == 1:
            # tol_range = np.geomspace(self.tol_level, 0.2, num=5)
            tol_range = np.array([0.2, 0.1, 1e-2, 1e-3, 1e-4])
            tol_budget = tol_range[index_fidelity]
        elif self.mf == 0:
            min_tol = -np.log(0.2)
            max_tol = -np.log(self.tol_level)
            tol_res = min_tol + index_fidelity*(max_tol - min_tol)
            tol_budget = np.exp(-tol_res)
        else:
            raise ValueError(
                "Please select one of two mf options continuous_fidelity or discrete_fidelity.")

        scaled_x = self.scale_domain(input_config)

        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=False)
        model = WeightedLasso(estimator=estimator)
        monitor = Monitor()
        sub_criterion = HeldOutMSE(None, None)
        criterion = CrossVal(sub_criterion, cv=self.kf)
        val_loss = criterion.get_val(model, self.X_train, self.y_train,
                                     log_alpha=scaled_x,
                                     monitor=monitor, tol=tol_budget)

        return val_loss/self.loss_oracle

    def test(self, input_config):
        """
        Post-processing metrics MSPE and Fscore

        Parameters
        ----------
        input_config : array size of n_features

        Returns
        -------
        MSPE divided by oracle and Fscore: list
        """

        if np.any(input_config < -1) or np.any(input_config > 1):
            raise ValueError(
                "The configuration is outside the bounds.")

        scaled_x = self.scale_domain(input_config)
        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=False)
        estimator.weights = np.exp(scaled_x)
        estimator.fit(self.X_train, self.y_train)
        self.reg_coef = estimator.coef_
        coef_hpo_support = np.abs(estimator.coef_) > self.eps_support
        fscore = f1_score(self.coef_true_support, coef_hpo_support)
        mspe = mean_squared_error(estimator.predict(self.X_test), self.y_test)
        mspe_div = mspe/self.mspe_oracle

        return({
                'mspe': mspe_div,  # scaled loss on test dataset
                'fscore': fscore  # Fscore for support recovery
                })

    def run_LASSOCV(self, n_alphas=100):
        """
        Running baseline LASSOCV

        Parameters
        ----------
        n_alphas : int, optional
            The number of grid points in 1D optimization (default is 100)

        Returns
        -------
        Cross-validation Loss, MSPE divided by oracle, fscore and time elapsed: list
        """

        # default number of alphas
        alphas = np.geomspace(self.alpha_max, self.alpha_min, n_alphas)

        lasso_params = dict(fit_intercept=False, tol=self.tol_level,
                            cv=self.kf, n_jobs=self.n_splits)

        # run LassoCV celer
        t0 = timeit.default_timer()
        model_lcv = LassoCV(alphas=alphas, **lasso_params)
        model_lcv.fit(self.X_train, self.y_train)
        t1 = timeit.default_timer()
        elapsed = t1 - t0

        min_lcv = np.where(model_lcv.mse_path_ == np.min(model_lcv.mse_path_))
        loss_lcv = np.mean(model_lcv.mse_path_[min_lcv[0]])

        mspe_lcv = mean_squared_error(
            model_lcv.predict(self.X_test), self.y_test)

        coef_lcv_support = np.abs(model_lcv.coef_) > self.eps_support
        fscore = f1_score(self.coef_true_support, coef_lcv_support)

        return({
                'val_loss': loss_lcv/self.loss_oracle,  # scaled validation loss
                'mspe': mspe_lcv/self.mspe_oracle, # scaled loss on test dataset
                'fscore': fscore,  # Fscore for support recovery
                'time': elapsed # Time elapsed wall-clock
                })

    def run_sparseho(self, grad_solver='gd', algo_pick='imp_forw', n_steps=10, init_point=None):
        """
        Running baseline Sparse-HO

        Parameters
        ----------
        grad_solver : str, optional
            Selecting which gradient solver to use gradient descent 'gd', 'adam' or 'line' as line search (default is gd)
        algo_pick   : str, optional
            Selecting which diff solver to use imp_forw or imp (default is imp_forw)
        n_steps     : int, optional
            Number of optimization steps (default is 10)
        init_point  : array, optional
            First guess (default is None)

        Returns
        -------
        Cross-validation loss, MSPE divided by oracle, fscore and time elapsed: list
        """

        if init_point is not None:
            init_point_scale = self.scale_domain(init_point)

        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=True)
        model = WeightedLasso(estimator=estimator)
        sub_criterion = HeldOutMSE(None, None)
        criterion = CrossVal(sub_criterion, cv=self.kf)
        if algo_pick == 'imp_forw':
            algo = ImplicitForward()
        elif algo_pick == 'imp':
            algo = Implicit()
        else:
            raise ValueError("Undefined algo_pick.")

        monitor = Monitor()
        if grad_solver == 'gd':
            optimizer = GradientDescent(n_outer=n_steps, tol=self.tol_level,
                                        verbose=False, p_grad_norm=1.9)
        elif grad_solver == 'adam':
            optimizer = Adam(n_outer=n_steps, lr=0.11, verbose=False, tol=self.tol_level)
        elif grad_solver == 'line':
            optimizer = LineSearch(n_outer=n_steps, verbose=False, tol=self.tol_level)

        if init_point is None:
            grad_search(
                algo, criterion, model, optimizer, self.X_train, self.y_train,
                self.alpha_max/10*np.ones((self.X_train.shape[1],)), monitor)
        else:
            grad_search(
                algo, criterion, model, optimizer, self.X_train, self.y_train,
                np.exp(init_point_scale), monitor)

        mspe = np.empty((n_steps,))
        fscore = np.empty((n_steps,))
        config_all = np.empty((n_steps, self.n_features))
        reg_coef = np.empty((n_steps, self.n_features))

        for i in range(n_steps):
            estimator.weights = monitor.alphas[i]
            config_all[i, :] = monitor.alphas[i]
            estimator.fit(self.X_train, self.y_train)
            mspe[i] = mean_squared_error(estimator.predict(self.X_test), self.y_test)
            reg_coef[i, :] = estimator.coef_
            coef_sho_support = np.abs(estimator.coef_) > self.eps_support
            fscore[i] = f1_score(self.coef_true_support, coef_sho_support)

        self.reg_coef = reg_coef
        self.config_all = config_all

        return({
                'val_loss': monitor.objs/self.loss_oracle,  # scaled validation loss
                'mspe': mspe/self.mspe_oracle, # scaled loss on test dataset
                'fscore': fscore,  # Fscore for support recovery
                'time': monitor.times # Time elapsed wall-clock
                })

class RealBenchmark():
    """
    Creating a real-world benchmark for a HPO algorithm.

    ...

    Attributes
    ----------
    pick_data : str
        name of dataset such as
        Diabetes, Breast_cancer, Leukemia, RCV1
    mf_opt : str, optional
        name of a multi-fidelity framework
        continuous_fidelity or discrete_fidelity
    tol_level: int, optional
        tolerance level for inner opt part
    n_splits: int, optional
        number of splits in CV
    test_size: int, optional
        percentage of test data
    seed: int, optional
        seed number

    Methods
    -------
    evaluate(input_config):
        Return cross-validation loss for configuration.
    fidelity_evaluate(input_config, index_fidelity=None):
        Return cross-validation loss for configuration and fidelity index.
    test(input_config):
        Return post-processing metric MSPE.
    run_LASSOCV(n_alphas=100):
        Running baseline LASSOCV and return loss, MSPE and time elapsed.
    run_sparseho(grad_solver='gd', algo_pick='imp_forw', n_steps=10, init_point=None, verbose=False):
        Running basedline Sparse-HO and return loss, MSPE and time elapsed.
    """
    def __init__(self, pick_data=None, mf_opt=None, tol_level=1e-4, n_splits=5, test_size=0.15, seed=42):
        """
        Constructs all the necessary attributes for real-world bench.

        Parameters
        ----------
            pick_data : str
                name of dataset such as
                Diabetes, Breast_cancer, DNA, Leukemia, RCV1
            mf_opt : str, optional
                name of a multi-fidelity framework
                continuous_fidelity or discrete_fidelity
            tol_level: int, optional
                tolerance level for inner opt part
            n_splits: int, optional
                number of splits in CV
            test_size: int, optional
                percentage of test data
            seed: int, optional
                seed number
        """

        self.tol_level = tol_level

        if pick_data.lower() == 'diabetes':
            X, y = fetch_libsvm('diabetes_scale')
            alpha_scale = 1e5
        elif pick_data.lower() == 'breast_cancer':
            X, y = fetch_libsvm('breast-cancer_scale')
            alpha_scale = 1e5
        elif pick_data.lower() == 'leukemia':
            self.X_train, self.y_train = fetch_libsvm(pick_data)
            self.X_test, self.y_test = fetch_libsvm('leukemia_test')
            alpha_scale = 1e5
        elif pick_data.lower() == 'rcv1':
            X, y = fetch_libsvm('rcv1.binary')
            alpha_scale = 1e3
        elif pick_data.lower() == 'dna':
            X, y = fetch_libsvm('dna')
            alpha_scale = 1e5
        else:
            raise ValueError("Unsupported dataset %s" % pick_data)

        self.mf = 2

        if mf_opt is not None:
            if mf_opt == 'continuous_fidelity':
                self.mf = 0
            elif mf_opt == 'discrete_fidelity':
                self.mf = 1
            else:
                raise ValueError(
                    "Please select one of two mf options continuous or discrete_fidelity.")

        # split train and test
        self.n_splits = n_splits
        if pick_data.lower() != 'leukemia':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed)

        self.n_features = self.X_train.shape[1]

        self.kf = KFold(shuffle=True, n_splits=n_splits, random_state=seed)

        self.alpha_max = np.max(np.abs(
            self.X_train.T @ self.y_train)) / len(self.y_train)
        self.alpha_min = self.alpha_max / alpha_scale

        self.log_alpha_min = np.log(self.alpha_min)
        self.log_alpha_max = np.log(self.alpha_max)

    def scale_domain(self, x):
        """Scaling the input configuration to the original Lasso space

        Args:
            x (numpy array): the input configuration scaled to [-1, 1]

        Returns:
           x_copy (numpy array): the original Lasso input configuration
        """
        x_copy = np.copy(x)
        x_copy = x_copy * (
                self.log_alpha_max - self.log_alpha_min) / 2 + (
                    self.log_alpha_max + self.log_alpha_min) / 2
        return x_copy

    def evaluate(self, input_config):
        """
        Evaluate configuration for synt bench

        Parameters
        ----------
        input_config : array size of n_features

        Returns
        -------
        Cross-validation Loss
        """

        if np.any(input_config < -1) or np.any(input_config > 1):
            raise ValueError(
                "The configuration is outside the bounds.")


        scaled_x = self.scale_domain(input_config)

        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=False)
        model = WeightedLasso(estimator=estimator)
        monitor = Monitor()
        sub_criterion = HeldOutMSE(None, None)
        criterion = CrossVal(sub_criterion, cv=self.kf)
        val_loss = criterion.get_val(model, self.X_train, self.y_train,
                                     log_alpha=scaled_x,
                                     monitor=monitor, tol=self.tol_level)

        return val_loss

    def fidelity_evaluate(self, input_config, index_fidelity=None):
        """
        Return cross-validation loss for selected fidelity.

        Parameters
        ----------
        input_config : array size of n_features
        index_fidelity : int, optional
            If mf_opt is selected, then selecting which fidelity to evaluate. (default is None)
            For continuous_fidelity, index_fidelity is defined within [0,1]
            For discrete_fidelity, index_fidelity is a dicreate par between 0 and 5.

        Returns
        -------
        Cross-validation Loss
        """

        if self.mf == 1:
            # tol_range = np.geomspace(self.tol_level, 0.2, num=5)
            tol_range = np.array([0.2, 0.1, 1e-2, 1e-3, 1e-4])
            tol_budget = tol_range[index_fidelity]
        elif self.mf == 0:
            min_tol = -np.log(0.2)
            max_tol = -np.log(self.tol_level)
            tol_res = min_tol + index_fidelity*(max_tol - min_tol)
            tol_budget = np.exp(-tol_res)
        else:
            raise ValueError(
                "Please select one of two mf options continuous_fidelity or discrete_fidelity.")

        scaled_x = self.scale_domain(input_config)

        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=False)
        model = WeightedLasso(estimator=estimator)
        monitor = Monitor()
        sub_criterion = HeldOutMSE(None, None)
        criterion = CrossVal(sub_criterion, cv=self.kf)
        val_loss = criterion.get_val(model, self.X_train, self.y_train,
                                     log_alpha=scaled_x,
                                     monitor=monitor, tol=tol_budget)

        return val_loss

    def test(self, input_config):
        """
        Post-processing metrics MSPE and Fscore

        Parameters
        ----------
        input_config : array size of n_features

        Returns
        -------
        MSPE
        """
        if np.any(input_config < -1) or np.any(input_config > 1):
            raise ValueError(
                "The configuration is outside the bounds.")

        scaled_x = self.scale_domain(input_config)
        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=False)
        estimator.weights = np.exp(scaled_x)
        estimator.fit(self.X_train, self.y_train)
        self.reg_coef = estimator.coef_
        mspe = mean_squared_error(estimator.predict(self.X_test), self.y_test)

        return({
                'mspe': mspe,  # loss on test dataset
                })

    def run_LASSOCV(self, n_alphas=100):
        """
        Running baseline LASSOCV

        Parameters
        ----------
        n_alphas : int, optional
            The number of grid points in 1D optimization (default is 100)

        Returns
        -------
        Cross-validation Loss, MSPE divided by oracle and time elapsed
        """

        # default number of alphas
        alphas = np.geomspace(self.alpha_max, self.alpha_min, n_alphas)

        lasso_params = dict(fit_intercept=False, tol=self.tol_level,
                            cv=self.kf, n_jobs=self.n_splits)

        # run LassoCV celer
        t0 = timeit.default_timer()
        model_lcv = LassoCV(alphas=alphas, **lasso_params)
        model_lcv.fit(self.X_train, self.y_train)
        t1 = timeit.default_timer()
        elapsed = t1 - t0
        # self.reg = model_lcv.coef_
        min_lcv = np.where(model_lcv.mse_path_ == np.min(model_lcv.mse_path_))
        loss_lcv = np.mean(model_lcv.mse_path_[min_lcv[0]])
        # self.minal = alphas[min_lcv[0]]
        mspe_lcv = mean_squared_error(
            model_lcv.predict(self.X_test), self.y_test)

        return({
                'val_loss': loss_lcv,  # validation loss
                'mspe': mspe_lcv, # loss on test dataset
                'time': elapsed # Time elapsed wall-clock
                })

    def run_sparseho(self, grad_solver='gd', algo_pick='imp_forw', n_steps=10, init_point=None):
        """
        Running baseline Sparse-HO

        Parameters
        ----------
        grad_solver : str, optional
            Selecting which gradient solver to use gradient descent 'gd', 'adam' or 'line' as line search (default is gd)
        algo_pick   : str, optional
            Selecting which diff solver to use imp_forw or imp (default is imp_forw).
            If Sparse-HO generates nan for a real-world bench such as 'rcv1',
            change the default algo_pick to 'imp'.
        n_steps     : int, optional
            Number of optimization steps (default is 10)
        init_point  : array, optional
            First guess (default is None)

        Returns
        -------
        Cross-validation loss, MSPE divided by oracle and time elapsed: list
        """

        if init_point is not None:
            init_point_scale = self.scale_domain(init_point)

        estimator = Lasso(fit_intercept=False, max_iter=100, warm_start=True)
        model = WeightedLasso(estimator=estimator)
        sub_criterion = HeldOutMSE(None, None)
        criterion = CrossVal(sub_criterion, cv=self.kf)
        if algo_pick == 'imp_forw':
            algo = ImplicitForward()
        elif algo_pick == 'imp':
            algo = Implicit()
        else:
            raise ValueError("Undefined algo_pick.")

        monitor = Monitor()
        if grad_solver == 'gd':
            optimizer = GradientDescent(n_outer=n_steps, tol=self.tol_level,
                                        verbose=False, p_grad_norm=1.9)
        elif grad_solver == 'adam':
            optimizer = Adam(n_outer=n_steps, lr=0.11, verbose=False, tol=self.tol_level)
        elif grad_solver == 'line':
            optimizer = LineSearch(n_outer=n_steps, verbose=False, tol=self.tol_level)

        if init_point is None:
            grad_search(
                algo, criterion, model, optimizer, self.X_train, self.y_train,
                self.alpha_max/10*np.ones((self.X_train.shape[1],)), monitor)
        else:
            grad_search(
                algo, criterion, model, optimizer, self.X_train, self.y_train,
                np.exp(init_point_scale), monitor)

        mspe = np.empty((n_steps,))
        config_all = np.empty((n_steps, self.n_features))
        reg_coef = np.empty((n_steps, self.n_features))

        for i in range(n_steps):
            estimator.weights = monitor.alphas[i]
            config_all[i, :] = monitor.alphas[i]
            estimator.fit(self.X_train, self.y_train)
            mspe[i] = mean_squared_error(estimator.predict(self.X_test), self.y_test)
            reg_coef[i, :] = estimator.coef_

        self.reg_coef = reg_coef
        self.config_all = config_all

        return({
                'val_loss': monitor.objs,  # validation loss
                'mspe': mspe, # loss on test dataset
                'time': monitor.times # Time elapsed wall-clock
                })
