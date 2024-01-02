import numpy as np
import xgboost as xgb

from collections import OrderedDict
from .ebo_core.helper import ConstantOffsetFn, NormalizedInputFn

from .push_function import PushReward
from .lunar_lander import LunarLander, heuristic_turbo
from .rover_function import create_small_domain
from .bipedal_walker import BipedalWalker, heuristic_bipedal
from sklearn import preprocessing
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, \
    precision_score, f1_score, log_loss
from .hpobench.dependencies.ml.data_manager import OpenMLDataManager
from .hpobench import config_file

class XGBoost_OpenML_Task:
    '''
    MLP_OpenML_Task: function

    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, task_id, bounds=None, sd=None, seed=0):
        self.input_dim = 9
        
        # Get training and validation datasets
        # Use the DataManager of the HPOBenchmark package
        data_path = config_file.data_dir / "OpenML"

        # Task ID is passed in by user
        valid_size = 0.33
        global_seed = 1
        dm = OpenMLDataManager(task_id, valid_size, data_path, global_seed)
        dm.load()

        train_X = dm.train_X
        valid_X = dm.valid_X

        train_y = dm.train_y
        valid_y = dm.valid_y
        
        # Convert to the proper format
        le = preprocessing.LabelEncoder()
        le.fit(list(train_y) + list(valid_y))

        train_y = le.transform(train_y)
        valid_y = le.transform(valid_y)

        self.X_train = train_X
        self.Y_train = train_y
        self.X_test = valid_X
        self.Y_test = valid_y
        self.num_classes = len(set(list(train_y) + list(valid_y)))

        # Tune 4 hyperparameters as in the HPOBench benchmark
        if bounds is None:
            # self.bounds = OrderedDict([('eta', (-10, 0)), 
            #                            ('max_depth', (0, 5.6439)),
            #                            ('colsample_bytree', (0.1, 1)),
            #                            ('reg_lambda', (-10, 10))])
            self.bounds = OrderedDict([('eta', (0, 1)),
                                       ('gamma', (0, 1)),
                                       ('max_depth', (0, 1)),
                                       ('min_child_weight', (0, 1)),
                                       ('max_delta_step', (0, 1)),
                                       ('colsample_bytree', (0, 1)),
                                       ('colsample_bylevel', (0, 1)),
                                       ('colsample_bynode', (0, 1)),
                                       ('reg_lambda', (0, 1))])
        else:
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = 1
        self.ismax = 1
        self.name = f'xgb-openml-task{task_id}'
        self.seed = seed

    def run_XGBoost(self, params):
        # NOTE: params has len being 1

        # Extract hyperparameters from params:
        params = params.ravel()
        
        # Transform the hyperparameters
        params_transform = params.copy()
        params_transform[0] = 10*params_transform[0] - 10 # eta
        params_transform[1] = 5.6439*params_transform[1] # gamma
        params_transform[2] = 5.6439*params_transform[2] # max_depth
        params_transform[3] = 5.6439*params_transform[3] # min_child_weight
        params_transform[4] = 5.6439*params_transform[4] # max_delta_step
        params_transform[5] = 0.9*params_transform[5] + 0.1 # colsample_bytree
        params_transform[6] = 0.9*params_transform[6] + 0.1 # colsample_bylevel
        params_transform[7] = 0.9*params_transform[7] + 0.1 # colsample_bynode
        params_transform[8] = 20*params_transform[8] - 10 # reg_lambda
        
        eta = 2**params[0]
        gamma = int(round(2**params[1]))
        max_depth = int(round(2**params[2]))
        min_child_weight = int(round(2**params[3]))
        max_delta_step = int(round(2**params[4]))
        colsample_bytree = params[5]
        colsample_bylevel = params[6]
        colsample_bynode = params[7]
        reg_lambda = 2**params[8]

        extra_args = dict(
            booster="gbtree",
            n_estimators=2000,
            objective="binary:logistic",
            random_state=None,
            subsample=1
        )

        if self.num_classes > 2:
            extra_args["objective"] = "multi:softmax"
            extra_args.update({"num_class": self.num_classes})

        model = xgb.XGBClassifier(
            learning_rate=eta,
            gamma=gamma,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            reg_lambda=reg_lambda,
            **extra_args
        )

        model.fit(self.X_train, self.Y_train)

        # Compute validation scores
        metrics = dict(
            acc=accuracy_score,
            bal_acc=balanced_accuracy_score,
            f1=f1_score,
            precision=precision_score,
            neglogloss=log_loss,
        )
        
        metrics_kwargs = dict(
            acc=dict(),
            bal_acc=dict(),
            f1=dict(average="macro", zero_division=0),
            precision=dict(average="macro", zero_division=0),
            neglogloss=dict()
        )
        
        scorers = dict()
        for k, v in metrics.items():
            scorers[k] = make_scorer(v, **metrics_kwargs[k])

        val_scores = dict()
        for k, v in scorers.items():
            val_scores[k] = v(model, self.X_test, self.Y_test)
        accuracy_val = val_scores["acc"]
        logloss_val = -val_scores["neglogloss"]

        return accuracy_val*100

    def func(self, params):

        if (type(params) == list):
            metrics_accuracy = np.zeros((len(params), 1))
            for i in range(len(params)):
                params_single = params[i]
                accuracy_temp = self.run_XGBoost(params_single)
                metrics_accuracy[i, 0] = accuracy_temp

        elif (type(params) == np.ndarray):
            # import os
            # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
            params_single = params.copy()
            metrics_accuracy = self.run_XGBoost(params_single)
        else:
            print('Something wrong with params!')

        return -metrics_accuracy


