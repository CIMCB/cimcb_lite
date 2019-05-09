import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import ParameterGrid

class BaseCrossVal(ABC):
    """Base class for crossval: kfold."""
    
    @abstractmethod
    def __init__(self, model, X, Y, param_dict, folds=10, bootnum=100):
        self.model = model 
        self.X = X
        self.Y = Y
        self.param_dict = param_dict
        self.param_list = list(ParameterGrid(param_dict))
        self.folds = folds
        self.bootnum = bootnum
        self.num_param = len(param_dict)
    
    @abstractmethod
    def calc_ypred(self):
        """Calculates ypred full and ypred cv."""
        pass
    
    @abstractmethod
    def calc_stats(self):
        """Calculates binary statistics from ypred full and ypred cv."""
        pass
    
    @abstractmethod
    def run(self):
        """Runs all functions prior to plot."""
        pass
    
    @abstractmethod
    def plot(self):
        """Creates a R2/Q2 plot."""
        pass