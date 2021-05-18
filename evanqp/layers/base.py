import numpy as np
from gurobipy import GRB
from enum import Enum


class Bound(Enum):
    INT_ARITHMETIC = 0
    ZONO_ARITHMETIC = 1


class BaseLayer:

    def __init__(self, out_size, depth):
        self.vars = {'out': []}
        self.bounds = {'out': {'lb': np.array([]), 'ub': np.array([])}}
        self.zono_bounds = {}
        self.out_size = out_size
        self.depth = depth

    def add_vars(self, model):
        self.vars['out'] = [model.addVar(vtype=GRB.CONTINUOUS,
                                         lb=self.bounds['out']['lb'][i] if self.bounds['out']['lb'].size > 0 else -GRB.INFINITY,
                                         ub=self.bounds['out']['ub'][i] if self.bounds['out']['ub'].size > 0 else GRB.INFINITY)
                            for i in range(self.out_size)]

    def add_constr(self, model, p_layer):
        pass

    def compute_bounds(self, method, p_layer):
        pass
