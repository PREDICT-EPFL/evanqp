import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from gurobipy import GRB, LinExpr, Model

from evanqp import Box
from evanqp.layers import Bound, BaseLayer, InputLayer
from evanqp.zonotope import Zonotope


class QPLayer(BaseLayer):

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

        if not only_primal:
            self.vars['mu'] = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY) for _ in range(self.A.shape[0] + self.problem.parameter_size())]
            self.vars['lam'] = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY) for _ in range(self.F.shape[0])]
            self.vars['r'] = [model.addVar(vtype=GRB.BINARY) for _ in range(self.F.shape[0])]

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

    def compute_bounds(self, method, p_layer, time_limit=None):
        model = Model()
        model.setParam('OutputFlag', 0)
        if time_limit is not None:
            model.setParam('TimeLimit', time_limit)

        input_set = Box(p_layer.bounds['out']['lb'], p_layer.bounds['out']['ub'])
        input_layer = InputLayer(input_set)
        input_layer.compute_bounds(method)
        input_layer.add_vars(model)
        model.update()
        input_layer.add_constr(model)
        model.update()

        old_vars = self.vars
        self.vars = {'out': []}
        self.bounds['out']['lb'] = [-GRB.INFINITY for _ in range(self.out_size)]
        self.bounds['out']['ub'] = [GRB.INFINITY for _ in range(self.out_size)]
        self.add_vars(model)
        model.update()
        self.add_constr(model, input_layer)
        model.update()

        self.bounds['out']['lb'] = np.zeros(self.problem.variable_size())
        self.bounds['out']['ub'] = np.zeros(self.problem.variable_size())

        for i in range(self.out_size):
            model.setObjective(self.vars['out'][i], GRB.MINIMIZE)
            model.update()
            model.optimize()
            self.bounds['out']['lb'][i] = model.objBound

            model.setObjective(self.vars['out'][i], GRB.MAXIMIZE)
            model.update()
            model.optimize()
            self.bounds['out']['ub'][i] = model.objBound

        self.vars = old_vars

        if method == Bound.ZONO_ARITHMETIC:
            self.zono_bounds['out'] = Zonotope.zonotope_from_box(self.bounds['out']['lb'], self.bounds['out']['ub'])

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
