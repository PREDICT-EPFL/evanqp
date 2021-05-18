import types
import math
import cvxpy as cp
import scipy.sparse as sp
from gurobipy import GRB, Model, LinExpr, abs_, max_
from copy import copy

from evanqp import Polytope
from evanqp.problems import QPProblem, MPCProblem
from evanqp.layers import Bound, InputLayer, QPLayer, SeqLayer


class Verifier:

    def __init__(self, parameter_set, *problems):
        self.input_layer = InputLayer(parameter_set)

        self.problems = []
        for problem in problems:
            if isinstance(problem, QPProblem):
                self.problems.append(QPLayer(problem, 1))
            else:
                self.problems.append(SeqLayer.from_pytorch(problem))

        self.bounds_calculated = False

    def compute_bounds(self, method=Bound.ZONO_ARITHMETIC):
        self.input_layer.compute_bounds(method)
        for p in self.problems:
            if isinstance(p, QPLayer):
                # for singular QPLayer bounds are not needed
                continue
            p.compute_bounds(method, self.input_layer)
        self.bounds_calculated = True

        return [p.bounds['out'] for p in self.problems]

    def setup_milp(self, model):
        if not self.bounds_calculated:
            self.compute_bounds()

        self.input_layer.add_vars(model)
        for p in self.problems:
            p.add_vars(model)
        model.update()

        self.input_layer.add_constr(model)
        for p in self.problems:
            p.add_constr(model, self.input_layer)
        model.update()

    def find_max_abs_diff(self, threads=0, output_flag=1):
        model = Model()
        model.setParam('OutputFlag', output_flag)
        model.setParam('Threads', threads)

        if len(self.problems) != 2:
            raise Exception('Number of problems must be 2.')
        if self.problems[0].out_size != self.problems[0].out_size:
            raise Exception('Problems do not have the same output size')

        self.setup_milp(model)

        diff = [model.addVar(vtype=GRB.CONTINUOUS) for _ in range(self.problems[0].out_size)]
        abs_diff = [model.addVar(vtype=GRB.CONTINUOUS) for _ in range(self.problems[0].out_size)]
        for i in range(self.problems[0].out_size):
            model.addConstr(diff[i] == self.problems[0].vars['out'][i] - self.problems[1].vars['out'][i])
            model.addConstr(abs_diff[i] == abs_(diff[i]))
        max_abs_diff = model.addVar(vtype=GRB.CONTINUOUS)
        model.addConstr(max_abs_diff == max_(abs_diff))

        model.setObjective(max_abs_diff, GRB.MAXIMIZE)
        model.update()
        model.optimize()

        return model.objBound, [p.x for p in self.input_layer.vars['out']]

    def verify_stability(self, threads=0, output_flag=1):
        if len(self.problems) != 2:
            raise Exception('Number of problems must be 2.')
        if not isinstance(self.problems[0], QPLayer) or not isinstance(self.problems[0].problem, MPCProblem):
            raise Exception('The first problem must be of type MPCProblem.')

        mpc_problem = self.problems[0].problem

        model = Model()
        model.setParam('OutputFlag', output_flag)
        model.setParam('Threads', threads)
        model.setParam('NonConvex', 2)  # allow non-convex MIQP formulation

        if not self.bounds_calculated:
            self.compute_bounds()

        reduced_objective_problem = copy(mpc_problem)

        # monkey patch original mpc problem with reduced objective function
        def problem_patch(_self):
            return cp.Problem(_self.reduced_objective(), _self.original_problem().constraints)
        reduced_objective_problem.original_problem = reduced_objective_problem.problem
        reduced_objective_problem.problem = types.MethodType(problem_patch, reduced_objective_problem)

        reduced_objective_mpc_layer = QPLayer(reduced_objective_problem, 1)
        reduced_objective_mpc_layer.compute_bounds(Bound.INT_ARITHMETIC, self.input_layer)

        self.input_layer.add_vars(model)
        self.problems[0].add_vars(model, only_primal=True)
        reduced_objective_mpc_layer.add_vars(model)
        self.problems[1].add_vars(model)
        model.update()

        self.input_layer.add_constr(model)
        self.problems[0].add_constr(model, self.input_layer, only_primal=True)
        reduced_objective_mpc_layer.add_constr(model, self.input_layer)
        self.problems[1].add_constr(model, self.input_layer)
        model.update()

        for i in range(reduced_objective_mpc_layer.out_size):
            model.addConstr(reduced_objective_mpc_layer.vars['out'][i] == self.problems[1].vars['out'][i])
        model.update()

        x = self.problems[0].vars['x']
        x_t = reduced_objective_mpc_layer.vars['x']

        obj = 0
        P_row_idx, P_col_idx, P_col_coef = sp.find(self.problems[0].P)
        for i, j, Pij in zip(P_row_idx, P_col_idx, P_col_coef):
            obj += 0.5 * x[i] * Pij * x[j]
        obj += LinExpr(self.problems[0].q, x)
        P_t_row_idx, P_t_col_idx, P_t_col_coef = sp.find(reduced_objective_mpc_layer.P)
        for i, j, Pij in zip(P_t_row_idx, P_t_col_idx, P_t_col_coef):
            obj -= 0.5 * x_t[i] * Pij * x_t[j]
        obj -= LinExpr(reduced_objective_mpc_layer.q, x_t)

        model.setObjective(obj, GRB.MINIMIZE)
        model.update()
        model.optimize()

        return model.objBound, [p.x for p in self.input_layer.vars['out']]

    def variables_in_polytope(self, poly, eps=1e-6, threads=0, output_flag=1):
        if len(self.problems) != 1:
            raise Exception('Number of problems must be 1.')
        if not isinstance(poly, Polytope):
            raise Exception('poly must be of type Polytope.')
        if self.problems[0].out_size != poly.A.shape[1]:
            raise Exception('poly shape does not match problem output size.')

        model = Model()
        model.setParam('OutputFlag', output_flag)
        model.setParam('Threads', threads)

        self.setup_milp(model)

        for i in range(self.problems[0].out_size):
            model.setObjective(LinExpr(poly.A[i, :], self.problems[0].vars['out']) - poly.b[i], GRB.MAXIMIZE)
            model.update()
            model.optimize()

            if model.objBound > eps:
                return False

        return True

    @staticmethod
    def min_optimal_mpc_horizon(parameter_set, mpc_factory, poly, eps=1e-6, threads=0):
        N = 1
        mpc_problem = mpc_factory(N)
        print(f'Checking N = {N}')
        verifier = Verifier(parameter_set, mpc_problem)
        res = verifier.variables_in_polytope(poly, eps=eps, threads=threads, output_flag=0)
        if res:
            return N

        lb = N + 1
        ub = float('inf')
        while lb < ub:
            if ub == float('inf'):
                N *= 2
            else:
                N = math.floor((lb + ub) / 2)
            mpc_problem = mpc_factory(N)
            print(f'Checking N = {N}')
            verifier = Verifier(parameter_set, mpc_problem)
            res = verifier.variables_in_polytope(poly, eps=eps, threads=threads, output_flag=0)
            if res:
                ub = N
            else:
                lb = N + 1
        return lb
