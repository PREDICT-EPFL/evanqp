import warnings
import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from gurobipy import GRB, LinExpr, Model
from tqdm import trange

from evanqp import Box
from evanqp.layers import BoundArithmetic, BaseLayer, ConstLayer, InputLayer
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

        self.lb = None
        self.ub = None
        self.big_m_ineq = None
        self.big_m_lam = None

    def add_vars(self, model, only_primal=False):
        super().add_vars(model)

        self.vars['x'] = [model.addVar(vtype=GRB.CONTINUOUS, lb=self.lb[i], ub=self.ub[i]) for i in range(self.P.shape[1])]

        if not only_primal:
            self.vars['mu'] = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY) for _ in range(self.A.shape[0] + self.problem.parameter_size())]
            self.vars['lam'] = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=self.big_m_lam[i]) for i in range(self.F.shape[0])]
            self.vars['r'] = [model.addVar(vtype=GRB.BINARY) for _ in range(self.F.shape[0])]
        else:
            if 'mu' in self.vars:
                del self.vars['mu']
            if 'lam' in self.vars:
                del self.vars['lam']
            if 'r' in self.vars:
                del self.vars['r']

    def add_constr(self, model, p_layer, only_primal=False, without_compl=False):
        # add previous layer as equality constraint
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

        # P @ x + q + A.T @ mu + F.T @ lam = 0
        if not only_primal:
            for i in range(self.P.shape[1]):
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
            model.addConstr(LinExpr(A_col_coef, [self.vars['x'][j] for j in A_col_idx]) == b_full[i])

        # F @ x <= g
        for i in range(self.F.shape[0]):
            _, F_col_idx, F_col_coef = sp.find(self.F[i, :])
            model.addConstr(LinExpr(F_col_coef, [self.vars['x'][j] for j in F_col_idx]) - self.g[i] <= 0)

        # (F_i @ x - g_i) * lam_i = 0
        if not only_primal and not without_compl:
            for i in range(self.F.shape[0]):
                _, F_col_idx, F_col_coef = sp.find(self.F[i, :])
                if self.big_m_ineq[i] < GRB.INFINITY:
                    model.addConstr(self.g[i] - LinExpr(F_col_coef, [self.vars['x'][j] for j in F_col_idx]) <= self.vars['r'][i] * self.big_m_ineq[i])
                model.addConstr((self.vars['r'][i] == 0) >> (LinExpr(F_col_coef, [self.vars['x'][j] for j in F_col_idx]) - self.g[i] == 0))

                if self.big_m_lam[i] < GRB.INFINITY:
                    model.addConstr(self.vars['lam'][i] <= (1 - self.vars['r'][i]) * self.big_m_lam[i])
                model.addConstr((self.vars['r'][i] == 1) >> (self.vars['lam'][i] == 0))

        i = 0
        for j in range(len(self.problem.variables())):
            idx = self.var_id_to_col[self.problem.variables()[j].id]
            size = self.problem.variables()[j].size
            for k in range(size):
                model.addConstr(self.vars['out'][i] == self.vars['x'][idx + k])
                i += 1

    def add_vars_jacobian(self, model, p_layer):
        self.vars['out_jac'] = np.empty((self.problem.variable_size(), self.problem.parameter_size()), dtype=object)
        for i in range(self.problem.variable_size()):
            for j in range(self.problem.parameter_size()):
                self.vars['out_jac'][i, j] = model.addVar(vtype=GRB.CONTINUOUS, lb=self.jacobian_bounds['out']['lb'][i, j], ub=self.jacobian_bounds['out']['ub'][i, j])

        self.vars['aux'] = []
        for i_aux in range(self.problem.parameter_size()):
            aux_vars = {}
            aux_vars['x'] = [model.addVar(vtype=GRB.CONTINUOUS, lb=self.aux_lb[i_aux][i], ub=self.aux_ub[i_aux][i]) for i in range(self.P.shape[1])]
            aux_vars['mu'] = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY) for _ in range(self.A.shape[0] + self.problem.parameter_size())]
            aux_vars['lam'] = [model.addVar(vtype=GRB.CONTINUOUS, lb=self.aux_big_m_lam_lb[i_aux][i], ub=self.aux_big_m_lam_ub[i_aux][i]) for i in range(self.F.shape[0])]
            aux_vars['out'] = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY) for _ in range(self.out_size)]
            self.vars['aux'].append(aux_vars)

    def add_constr_jacobian(self, model, p_layer, without_compl=False):
        if not isinstance(p_layer, InputLayer):
            raise Exception('Jacobains of QP layers have to first.')

        pertubation = 1.0

        for i_aux in range(self.problem.parameter_size()):
            # add previous layer as equality constraint
            A_par = np.zeros((self.problem.parameter_size(), self.A.shape[1]))
            b_par = np.zeros(self.problem.parameter_size(), dtype=object)
            i = 0
            for j in range(len(self.problem.parameters())):
                idx = self.var_id_to_col[self.problem.parameters()[j].id]
                size = self.problem.parameters()[j].size
                for k in range(size):
                    A_par[i, idx + k] = 1
                    b_par[i] = p_layer.vars['out'][i]
                    if i_aux == i:
                        b_par[i] += pertubation
                    i += 1

            A_full = sp.vstack((self.A, A_par), format='csc')
            b_full = np.concatenate((self.b, b_par))

            # P @ x + q + A.T @ mu + F.T @ lam = 0
            for i in range(self.P.shape[1]):
                _, P_col_idx, P_col_coef = sp.find(self.P[i, :])
                _, A_col_idx, A_col_coef = sp.find(A_full.T[i, :])
                _, F_col_idx, F_col_coef = sp.find(self.F.T[i, :])
                model.addConstr(LinExpr(P_col_coef, [self.vars['aux'][i_aux]['x'][j] for j in P_col_idx])
                                + self.q[i]
                                + LinExpr(A_col_coef, [self.vars['aux'][i_aux]['mu'][j] for j in A_col_idx])
                                + LinExpr(F_col_coef, [self.vars['aux'][i_aux]['lam'][j] for j in F_col_idx])
                                == 0)

            # A @ x = b
            for i in range(A_full.shape[0]):
                _, A_col_idx, A_col_coef = sp.find(A_full[i, :])
                model.addConstr(LinExpr(A_col_coef, [self.vars['aux'][i_aux]['x'][j] for j in A_col_idx]) == b_full[i])

            # (F_i @ x - g_i) * lam_i = 0
            if not without_compl:
                for i in range(self.F.shape[0]):
                    _, F_col_idx, F_col_coef = sp.find(self.F[i, :])
                    model.addConstr((self.vars['r'][i] == 0) >> (LinExpr(F_col_coef, [self.vars['aux'][i_aux]['x'][j] for j in F_col_idx]) - self.g[i] == 0))
                    if self.aux_big_m_ineq_lb[i_aux][i] > -GRB.INFINITY:
                        model.addConstr(self.g[i] - LinExpr(F_col_coef, [self.vars['aux'][i_aux]['x'][j] for j in F_col_idx]) >= self.vars['r'][i] * self.aux_big_m_ineq_lb[i_aux][i])
                    if self.aux_big_m_ineq_ub[i_aux][i] < GRB.INFINITY:
                        model.addConstr(self.g[i] - LinExpr(F_col_coef, [self.vars['aux'][i_aux]['x'][j] for j in F_col_idx]) <= self.vars['r'][i] * self.aux_big_m_ineq_ub[i_aux][i])

                    model.addConstr((self.vars['r'][i] == 1) >> (self.vars['aux'][i_aux]['lam'][i] == 0))
                    if self.aux_big_m_lam_lb[i_aux][i] > -GRB.INFINITY:
                        model.addConstr(self.vars['aux'][i_aux]['lam'][i] >= (1 - self.vars['r'][i]) * self.aux_big_m_lam_lb[i_aux][i])
                    if self.aux_big_m_lam_ub[i_aux][i] < GRB.INFINITY:
                        model.addConstr(self.vars['aux'][i_aux]['lam'][i] <= (1 - self.vars['r'][i]) * self.aux_big_m_lam_ub[i_aux][i])

            i = 0
            for j in range(len(self.problem.variables())):
                idx = self.var_id_to_col[self.problem.variables()[j].id]
                size = self.problem.variables()[j].size
                for k in range(size):
                    model.addConstr(self.vars['aux'][i_aux]['out'][i] == self.vars['aux'][i_aux]['x'][idx + k])
                    i += 1

        for i in range(self.problem.variable_size()):
            for j in range(self.problem.parameter_size()):
                model.addConstr(self.vars['out_jac'][i, j] == 1.0 / pertubation * (self.vars['aux'][j]['out'][i] - self.vars['out'][i]))

    def compute_bounds_jacobian(self, p_layer, dual_bound=1e6, mip=False, time_limit=1, with_output=True, **kwargs):
        model = Model()
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', time_limit)

        # add bounds from previous layer as constraints
        input_set = Box(p_layer.bounds['out']['lb'], p_layer.bounds['out']['ub'])
        input_layer = InputLayer(input_set)
        input_layer.compute_bounds(BoundArithmetic.INT_ARITHMETIC, **kwargs)
        input_layer.add_vars(model)
        model.update()
        input_layer.add_constr(model)
        model.update()

        old_vars = self.vars
        self.vars = {'out': []}

        self.jacobian_bounds['out'] = {}
        self.jacobian_bounds['out']['lb'] = -GRB.INFINITY * np.ones((self.problem.variable_size(), self.problem.parameter_size()))
        self.jacobian_bounds['out']['ub'] = GRB.INFINITY * np.ones((self.problem.variable_size(), self.problem.parameter_size()))

        self.aux_lb = [-GRB.INFINITY * np.ones(self.A.shape[1]) for _ in range(self.problem.parameter_size())]
        self.aux_ub = [GRB.INFINITY * np.ones(self.A.shape[1]) for _ in range(self.problem.parameter_size())]

        self.aux_big_m_ineq_lb = [-dual_bound * np.ones(self.F.shape[0]) for _ in range(self.problem.parameter_size())]
        self.aux_big_m_ineq_ub = [dual_bound * np.ones(self.F.shape[0]) for _ in range(self.problem.parameter_size())]
        self.aux_big_m_lam_lb = [-dual_bound * np.ones(self.F.shape[0]) for _ in range(self.problem.parameter_size())]
        self.aux_big_m_lam_ub = [dual_bound * np.ones(self.F.shape[0]) for _ in range(self.problem.parameter_size())]

        self.add_vars(model)
        model.update()
        self.add_constr(model, input_layer, without_compl=(not mip))
        model.update()

        self.add_vars_jacobian(model, input_layer)
        model.update()
        self.add_constr_jacobian(model, input_layer, without_compl=(not mip))
        model.update()

        for i in trange(self.F.shape[0], desc='Aux QP Ineq Bound'):
            for i_aux in range(self.problem.parameter_size()):
                _, F_col_idx, F_col_coef = sp.find(self.F[i, :])
                model.setObjective(self.g[i] - LinExpr(F_col_coef, [self.vars['aux'][i_aux]['x'][j] for j in F_col_idx]), GRB.MINIMIZE)
                model.update()
                model.optimize()
                if model.Status == GRB.OPTIMAL:
                    self.aux_big_m_ineq_lb[i_aux][i] = model.objBound

                model.setObjective(self.g[i] - LinExpr(F_col_coef, [self.vars['aux'][i_aux]['x'][j] for j in F_col_idx]), GRB.MAXIMIZE)
                model.update()
                model.optimize()
                if model.Status == GRB.OPTIMAL:
                    self.aux_big_m_ineq_ub[i_aux][i] = model.objBound

        for i in trange(self.F.shape[0], desc='Aux QP Dual Bound'):
            for i_aux in range(self.problem.parameter_size()):
                model.setObjective(self.vars['aux'][i_aux]['lam'][i], GRB.MINIMIZE)
                model.update()
                model.optimize()
                if model.Status == GRB.OPTIMAL:
                    self.aux_big_m_lam_lb[i_aux][i] = model.objBound

                model.setObjective(self.vars['aux'][i_aux]['lam'][i], GRB.MAXIMIZE)
                model.update()
                model.optimize()
                if model.Status == GRB.OPTIMAL:
                    self.aux_big_m_lam_ub[i_aux][i] = model.objBound

        if with_output:
            for i in trange(self.problem.variable_size(), desc='QP Gain Bound'):
                for j in range(self.problem.parameter_size()):
                    model.setObjective(self.vars['out_jac'][i, j], GRB.MINIMIZE)
                    model.update()
                    model.optimize()
                    if model.Status == GRB.OPTIMAL:
                        self.jacobian_bounds['out']['lb'][i, j] = model.objBound

                    model.setObjective(self.vars['out_jac'][i, j], GRB.MAXIMIZE)
                    model.update()
                    model.optimize()
                    if model.Status == GRB.OPTIMAL:
                        self.jacobian_bounds['out']['ub'][i, j] = model.objBound

        self.vars = old_vars

    def compute_bounds(self, method, p_layer, dual_bound=1e6, mip=False, time_limit=1, with_output=True,  **kwargs):
        model = Model()
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', time_limit)

        # add bounds from previous layer as constraints
        input_set = Box(p_layer.bounds['out']['lb'], p_layer.bounds['out']['ub'])
        input_layer = InputLayer(input_set)
        input_layer.compute_bounds(method, **kwargs)
        input_layer.add_vars(model)
        model.update()
        input_layer.add_constr(model)
        model.update()

        old_vars = self.vars
        self.vars = {'out': []}

        self.bounds['out']['lb'] = np.array([-GRB.INFINITY for _ in range(self.out_size)])
        self.bounds['out']['ub'] = np.array([GRB.INFINITY for _ in range(self.out_size)])

        self.lb = -GRB.INFINITY * np.ones(self.A.shape[1])
        self.ub = GRB.INFINITY * np.ones(self.A.shape[1])

        self.big_m_ineq = dual_bound * np.ones(self.F.shape[0])
        self.big_m_lam = dual_bound * np.ones(self.F.shape[0])

        self.add_vars(model)
        model.update()
        self.add_constr(model, input_layer, without_compl=(not mip))
        model.update()

        for i in trange(self.F.shape[0], desc='QP Ineq Bound'):
            _, F_col_idx, F_col_coef = sp.find(self.F[i, :])
            model.setObjective(self.g[i] - LinExpr(F_col_coef, [self.vars['x'][j] for j in F_col_idx]), GRB.MAXIMIZE)
            model.update()
            model.optimize()
            if model.Status == GRB.OPTIMAL:
                self.big_m_ineq[i] = model.objBound

        for i in trange(self.F.shape[0], desc='QP Dual Bound'):
            model.setObjective(self.vars['lam'][i], GRB.MAXIMIZE)
            model.update()
            model.optimize()
            if model.Status == GRB.OPTIMAL:
                self.big_m_lam[i] = model.objBound

        if with_output:
            i = 0
            for j in range(len(self.problem.variables())):
                idx = self.var_id_to_col[self.problem.variables()[j].id]
                size = self.problem.variables()[j].size

                for k in trange(size, desc='QP Output Bound'):
                    model.setObjective(self.vars['x'][idx + k], GRB.MINIMIZE)
                    model.update()
                    model.optimize()
                    if model.Status == GRB.OPTIMAL:
                        self.lb[idx + k] = model.objBound
                    if self.lb[idx + k] == -GRB.INFINITY:
                        warnings.warn('QP Problem output variables are not lower bounded. Consider adding explicit bounds.')

                    model.setObjective(self.vars['x'][idx + k], GRB.MAXIMIZE)
                    model.update()
                    model.optimize()
                    if model.Status == GRB.OPTIMAL:
                        self.ub[idx + k] = model.objBound
                    if self.ub[idx + k] == GRB.INFINITY:
                        warnings.warn('QP Problem output variables are not upper bounded. Consider adding explicit bounds.')

                self.bounds['out']['lb'][i:i+size] = self.lb[idx:idx+size]
                self.bounds['out']['ub'][i:i+size] = self.ub[idx:idx+size]
                i += size

        if method == BoundArithmetic.ZONO_ARITHMETIC:
            self.zono_bounds['out'] = Zonotope.zonotope_from_box(self.bounds['out']['lb'], self.bounds['out']['ub'])

        self.vars = old_vars

    @staticmethod
    def replace_params(expr, param_to_placeholder, placeholder_to_param):
        for idx, arg in enumerate(expr.args):
            if isinstance(arg, cp.Parameter):
                QPLayer.replace_param(expr, idx, param_to_placeholder, placeholder_to_param)
            else:
                QPLayer.replace_params(arg, param_to_placeholder, placeholder_to_param)

    @staticmethod
    def replace_param(expr, idx, param_to_placeholder, placeholder_to_param):
        param = expr.args[idx]
        if param.id not in param_to_placeholder:
            placeholder = cp.Variable(param.shape, var_id=param.id, name=param.name())
            param_to_placeholder[param.id] = placeholder
            placeholder_to_param[placeholder.id] = param
        expr.args[idx] = param_to_placeholder[param.id]

    @staticmethod
    def restore_params(expr, placeholder_to_param):
        for idx, arg in enumerate(expr.args):
            if isinstance(arg, cp.Variable) and arg.id in placeholder_to_param:
                expr.args[idx] = placeholder_to_param[arg.id]
            else:
                QPLayer.restore_params(arg, placeholder_to_param)

    @staticmethod
    def compile_qp_problem_with_params_as_variables(problem):
        objective = problem.objective
        constraints = problem.constraints

        param_to_placeholder = {}
        placeholder_to_param = {}
        QPLayer.replace_params(objective, param_to_placeholder, placeholder_to_param)
        for con in constraints:
            QPLayer.replace_params(con, param_to_placeholder, placeholder_to_param)
        problem = cp.Problem(objective, constraints)

        data, chain, inverse_data = problem.get_problem_data(cp.OSQP)
        compiler = data[cp.settings.PARAM_PROB]

        P = data['P']
        q = data['q']
        A = data['A']
        b = data['b']
        F = data['F']
        g = data['G']

        QPLayer.restore_params(objective, placeholder_to_param)
        for con in constraints:
            QPLayer.restore_params(con, placeholder_to_param)

        return P, q, A, b, F, g, compiler.var_id_to_col

    def forward(self, x, warm_start=False):
        # we can only warm start if MILP formulation
        warm_start = warm_start and ('r' in self.vars)

        model = Model()
        model.setParam('OutputFlag', 0)

        input_layer = ConstLayer(x, 0)
        input_layer.add_vars(model)
        model.update()
        input_layer.add_constr(model)
        model.update()

        original_vars = self.vars

        self.vars = {'out': []}
        self.add_vars(model, only_primal=True)
        model.update()
        self.add_constr(model, input_layer, only_primal=True)
        model.update()

        obj = 0
        x_var = self.vars['x']
        P_row_idx, P_col_idx, P_col_coef = sp.find(self.P)
        for i, j, Pij in zip(P_row_idx, P_col_idx, P_col_coef):
            obj += 0.5 * x_var[i] * Pij * x_var[j]
        obj += LinExpr(self.q, x_var)

        model.setObjective(obj, GRB.MINIMIZE)
        model.update()
        model.optimize()

        if warm_start:
            x = np.array([e.x for e in self.vars['x']])
            ineq = self.F @ x - self.g
            for i in range(self.F.shape[0]):
                if abs(ineq[i]) <= 1e-7:
                    original_vars['r'][i].Start = 0
                else:
                    original_vars['r'][i].Start = 1

        result = np.zeros(self.out_size)
        for i in range(self.out_size):
            result[i] = self.vars['out'][i].x

        self.vars = original_vars

        return result
