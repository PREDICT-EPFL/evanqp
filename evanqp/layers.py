import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
from gurobipy import GRB, LinExpr
from enum import Enum

from evanqp.sets import Box, Polytope
from evanqp.zonotope import Zonotope


class Bound(Enum):
    INT_ARITHMETIC = 0
    ZONO_ARITHMETIC = 1


class Layer:

    def __init__(self, out_size, depth):
        self.vars = {'out': []}
        self.bounds = {'out': {'lb': np.array([]), 'ub': np.array([])}}
        self.zono_bounds = {}
        self.out_size = out_size
        self.depth = depth

    def add_vars(self, model):
        self.vars['out'] = [model.addVar(vtype=GRB.CONTINUOUS,
                                         lb=self.bounds['out']['lb'][i],
                                         ub=self.bounds['out']['ub'][i])
                            for i in range(self.out_size)]

    def add_constr(self, model, p_layer):
        pass

    def compute_bounds(self, method, p_layer):
        pass


class InputLayer(Layer):

    def __init__(self, input_set):
        self.input_set = input_set
        if isinstance(self.input_set, Polytope):
            out_size = self.input_set.A.shape[1]
        elif isinstance(self.input_set, Box):
            out_size = self.input_set.lb.shape[0]
        else:
            raise RuntimeError('Set is not supported.')

        super().__init__(out_size, 0)

    def add_constr(self, model, p_layer=None):
        if isinstance(self.input_set, Polytope):
            A, b = self.input_set.A, self.input_set.b
            for i in range(A.shape[0]):
                model.addConstr(LinExpr(A[i, :], self.vars['out']) <= b[i])

    def compute_bounds(self, method, p_layer=None):
        if isinstance(self.input_set, Polytope):
            self.bounds['out']['lb'] = self.input_set.bounding_box().lb
            self.bounds['out']['ub'] = self.input_set.bounding_box().ub
        elif isinstance(self.input_set, Box):
            self.bounds['out']['lb'] = self.input_set.lb
            self.bounds['out']['ub'] = self.input_set.ub

        if method == Bound.ZONO_ARITHMETIC:
            self.zono_bounds['out'] = Zonotope.zonotope_from_box(self.bounds['out']['lb'], self.bounds['out']['ub'])


class LinearLayer(Layer):

    def __init__(self, weight, bias, depth):
        super().__init__(weight.shape[0], depth)

        self.weight = weight
        self.bias = bias

    def add_constr(self, model, p_layer):
        for i in range(self.out_size):
            model.addConstr(self.vars['out'][i] == LinExpr(self.weight[i, :], p_layer.vars['out']) + self.bias[i])

    def compute_bounds(self, method, p_layer):
        if method == Bound.INT_ARITHMETIC:
            self._compute_bounds_ia(p_layer)
        elif method == Bound.ZONO_ARITHMETIC:
            self._compute_bounds_ia(p_layer)
            self._compute_bounds_zono(p_layer)
        else:
            raise NotImplementedError(f'Method {method} not implemented.')

    def _compute_bounds_ia(self, p_layer):
        W_p = np.clip(self.weight, 0, None)
        W_m = np.clip(self.weight, None, 0)

        self.bounds['out']['lb'] = W_m @ p_layer.bounds['out']['ub'] + W_p @ p_layer.bounds['out']['lb'] + self.bias
        self.bounds['out']['ub'] = W_p @ p_layer.bounds['out']['ub'] + W_m @ p_layer.bounds['out']['lb'] + self.bias

    def _compute_bounds_zono(self, p_layer):
        self.zono_bounds['out'] = p_layer.zono_bounds['out'].linear(self.weight, self.bias)
        lb, ub = self.zono_bounds['out'].concretize()
        self.bounds['out']['lb'] = np.maximum(self.bounds['out']['lb'], lb)
        self.bounds['out']['ub'] = np.minimum(self.bounds['out']['ub'], ub)


class ReluLayer(Layer):

    def __init__(self, out_size, depth):
        super().__init__(out_size, depth)

    def add_vars(self, model, linear_approx=False):
        super().add_vars(model)

        self.vars['r'] = [model.addVar(vtype=(GRB.CONTINUOUS if linear_approx else GRB.BINARY),
                                       lb=0,
                                       ub=1)
                          for _ in range(self.out_size)]
        for i in range(self.out_size):
            self.vars['r'][i].setAttr(GRB.Attr.BranchPriority, self.depth)

    def add_constr(self, model, p_layer, linear_approx=False):
        for i in range(self.out_size):
            if p_layer.bounds['out']['lb'][i] >= 0:
                model.addConstr(self.vars['out'][i] == p_layer.vars['out'][i])
            elif p_layer.bounds['out']['ub'][i] <= 0:
                model.addConstr(self.vars['out'][i] == 0)
            else:
                model.addConstr(self.vars['out'][i] >= 0)
                model.addConstr(self.vars['out'][i] >= p_layer.vars['out'][i])
                model.addConstr(self.vars['out'][i] <= p_layer.bounds['out']['ub'][i] * self.vars['r'][i])
                model.addConstr(self.vars['out'][i] <= p_layer.vars['out'][i] - p_layer.bounds['out']['lb'][i] * (1 - self.vars['r'][i]))
                if not linear_approx:
                    model.addConstr((self.vars['r'][i] == 1) >> (p_layer.vars['out'][i] >= 0))
                    model.addConstr((self.vars['r'][i] == 0) >> (p_layer.vars['out'][i] <= 0))

    def compute_bounds(self, method, p_layer):
        if method == Bound.INT_ARITHMETIC:
            self._compute_bounds_ia(p_layer)
        elif method == Bound.ZONO_ARITHMETIC:
            self._compute_bounds_ia(p_layer)
            self._compute_bounds_zono(p_layer)
        else:
            raise NotImplementedError(f'Method {method} not implemented.')

    def _compute_bounds_ia(self, p_layer):
        self.bounds['out']['lb'] = np.clip(p_layer.bounds['out']['lb'], 0, None)
        self.bounds['out']['ub'] = np.clip(p_layer.bounds['out']['ub'], 0, None)

    def _compute_bounds_zono(self, p_layer):
        p_lb, p_ub = p_layer.bounds['out']['lb'], p_layer.bounds['out']['ub']
        self.zono_bounds['out'] = p_layer.zono_bounds['out'].relu(bounds=(p_lb, p_ub))
        lb, ub = self.zono_bounds['out'].concretize()
        self.bounds['out']['lb'] = np.maximum(self.bounds['out']['lb'], lb)
        self.bounds['out']['ub'] = np.minimum(self.bounds['out']['ub'], ub)


class QPLayer(Layer):

    def __init__(self, problem, depth):
        super().__init__(problem.variable_size(), depth)

        self.problem = problem
        P, q, A, b, F, g, var_id_to_col = self.compile_qp_problem_with_params_as_variables(problem.problem())
        self.P = P
        self.q = q
        self.A = A
        self.b = b
        self.F = F
        self.g = g
        self.var_id_to_col = var_id_to_col

    def add_vars(self, model, only_primal=False):
        super().add_vars(model)

        self.vars['x'] = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY) for _ in range(self.P.shape[1])]
        for i in range(self.P.shape[1]):
            self.vars['x'][i].setAttr(GRB.Attr.BranchPriority, self.depth)

        if not only_primal:
            self.vars['mu'] = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY) for _ in range(self.A.shape[0] + self.problem.parameter_size())]
            for i in range(self.A.shape[0] + self.problem.parameter_size()):
                self.vars['mu'][i].setAttr(GRB.Attr.BranchPriority, self.depth)

            self.vars['lam'] = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY) for _ in range(self.F.shape[0])]
            for i in range(self.F.shape[0]):
                self.vars['lam'][i].setAttr(GRB.Attr.BranchPriority, self.depth)

            self.vars['r'] = [model.addVar(vtype=GRB.BINARY) for _ in range(self.F.shape[0])]
            for i in range(self.out_size):
                self.vars['r'][i].setAttr(GRB.Attr.BranchPriority, self.depth)

    def add_constr(self, model, p_layer, only_primal=False):
        A_par = np.zeros((self.problem.parameter_size(), self.A.shape[1]))
        b_par = np.zeros(self.problem.parameter_size(), dtype=object)
        i = 0
        for j in range(len(self.problem.parameters())):
            idx = self.var_id_to_col[self.problem.parameters()[j].id]
            size = self.problem.parameters()[j].size
            for k in range(size):
                A_par[i, idx + k] = 1
                b_par[i] = p_layer.vars['out'][i]
                i += 1

        A_full = sp.vstack((self.A, A_par), format='csc')
        b_full = np.concatenate((self.b, b_par))

        if not only_primal:
            # P @ x + q + A.T @ mu + F.T @ lam = 0
            for i in range(self.P.shape[0]):
                _, P_col_idx, P_col_coef = sp.find(self.P[i, :])
                _, A_col_idx, A_col_coef = sp.find(A_full.T[i, :])
                _, F_col_idx, F_col_coef = sp.find(self.F.T[i, :])
                model.addConstr(LinExpr(P_col_coef, [self.vars['x'][j] for j in P_col_idx])
                                + self.q[i]
                                + LinExpr(A_col_coef, [self.vars['mu'][j] for j in A_col_idx])
                                + LinExpr(F_col_coef, [self.vars['lam'][j] for j in F_col_idx])
                                == 0)

        # A @ x = b
        for i in range(A_full.shape[0]):
            _, A_col_idx, A_col_coef = sp.find(A_full[i, :])
            model.addConstr(LinExpr(A_col_coef, [self.vars['x'][j] for j in A_col_idx]) - b_full[i] == 0)
        # F @ x <= g
        for i in range(self.F.shape[0]):
            _, F_col_idx, F_col_coef = sp.find(self.F[i, :])
            model.addConstr(LinExpr(F_col_coef, [self.vars['x'][j] for j in F_col_idx]) - self.g[i] <= 0)

        if not only_primal:
            # (F_i @ x - g_i) * lam_i = 0
            for i in range(self.F.shape[0]):
                # M = 1e5
                # model.addConstr(F[i, :] @ x - g[i] >= -r[i] * M)
                # model.addConstr(lam[i] <= (1 - r[i]) * M)
                _, F_col_idx, F_col_coef = sp.find(self.F[i, :])
                model.addConstr((self.vars['r'][i] == 0) >> (LinExpr(F_col_coef, [self.vars['x'][j] for j in F_col_idx]) - self.g[i] == 0))
                model.addConstr((self.vars['r'][i] == 1) >> (self.vars['lam'][i] == 0))

        i = 0
        for j in range(len(self.problem.variables())):
            idx = self.var_id_to_col[self.problem.variables()[j].id]
            size = self.problem.variables()[j].size
            for k in range(size):
                model.addConstr(self.vars['out'][i] == self.vars['x'][idx + k])
                i += 1

    def compute_bounds(self, method, p_layer):
        self.bounds['out']['lb'] = np.zeros(self.problem.variable_size())
        self.bounds['out']['ub'] = np.zeros(self.problem.variable_size())

        x = cp.Variable(self.A.shape[1])

        constraints = [self.A @ x == self.b, self.F @ x <= self.g]
        i = 0
        for j in range(len(self.problem.parameters())):
            idx = self.var_id_to_col[self.problem.parameters()[j].id]
            size = self.problem.parameters()[j].size
            constraints += [p_layer.bounds['out']['lb'][i:i+size] <= x[idx:idx+size]]
            constraints += [x[idx:idx+size] <= p_layer.bounds['out']['ub'][i:i+size]]
            i += size

        i = 0
        for j in range(len(self.problem.variables())):
            idx = self.var_id_to_col[self.problem.variables()[j].id]
            size = self.problem.variables()[j].size
            for k in range(size):
                prob = cp.Problem(cp.Minimize(x[idx + k]), constraints)
                prob.solve()
                if prob.status in [cp.INFEASIBLE, cp.UNBOUNDED]:
                    raise Exception('Could not compute bounds for QP.')
                self.bounds['out']['lb'][i] = prob.value

                prob = cp.Problem(cp.Maximize(x[idx + k]), constraints)
                prob.solve()
                if prob.status in [cp.INFEASIBLE, cp.UNBOUNDED]:
                    raise Exception('Could not compute bounds for QP.')
                self.bounds['out']['ub'][i] = prob.value

                i += 1

    @staticmethod
    def replace_params(expr, param_dic):
        for idx, arg in enumerate(expr.args):
            if isinstance(arg, cp.Parameter):
                param_dic = QPLayer.replace_param(expr, idx, param_dic)
            else:
                param_dic = QPLayer.replace_params(arg, param_dic)
        return param_dic

    @staticmethod
    def replace_param(expr, idx, param_dic):
        param = expr.args[idx]
        placeholder = cp.Variable(param.shape, var_id=param.id, name=param.name())
        expr.args[idx] = placeholder
        param_dic[placeholder.id] = (expr, idx, param)
        return param_dic

    @staticmethod
    def restore_params(expr, param_dic):
        for idx, arg in enumerate(expr.args):
            if isinstance(arg, cp.Variable) and arg.id in param_dic:
                expr.args[idx] = param_dic[arg.id][2]
            else:
                QPLayer.restore_params(arg, param_dic)

    @staticmethod
    def compile_qp_problem_with_params_as_variables(problem):
        objective = problem.objective
        constraints = problem.constraints

        param_dic = {}
        QPLayer.replace_params(objective, param_dic)
        for con in constraints:
            QPLayer.replace_params(con, param_dic)
        problem = cp.Problem(objective, constraints)

        data, chain, inverse_data = problem.get_problem_data(cp.OSQP)
        compiler = data[cp.settings.PARAM_PROB]

        P = data['P']
        q = data['q']
        A = data['A']
        b = data['b']
        F = data['F']
        g = data['G']

        QPLayer.restore_params(objective, param_dic)
        for con in constraints:
            QPLayer.restore_params(con, param_dic)

        return P, q, A, b, F, g, compiler.var_id_to_col


class SeqLayers:

    def __init__(self, layers):
        self.layers = layers

    def add_vars(self, model):
        for layer in self.layers:
            layer.add_vars(model)

    def add_constr(self, model, p_layer):
        for layer in self.layers:
            layer.add_constr(model, p_layer)
            p_layer = layer

    def compute_bounds(self, method, p_layer):
        for layer in self.layers:
            layer.compute_bounds(method, p_layer)
            p_layer = layer

    @staticmethod
    def from_pytorch(pytorch_model, start_depth=1):
        if not isinstance(pytorch_model, nn.Sequential):
            pytorch_model = nn.Sequential(pytorch_model)

        layers = []
        for i, layer in enumerate(pytorch_model):
            if isinstance(layer, nn.Linear):
                weight = layer.weight.detach().cpu().double().numpy()
                bias = layer.bias.detach().cpu().double().numpy()
                layers.append(LinearLayer(weight, bias, i + start_depth))
            elif isinstance(layer, nn.ReLU):
                out_size = layers[i - 1].out_size
                layers.append(ReluLayer(out_size, i + start_depth))
            else:
                raise NotImplementedError(f'Pytorch Layer {layer} not supported.')

        return SeqLayers(layers)
