import types
import math
import numpy as np
import cvxpy as cp
import scipy.sparse as sp
from gurobipy import GRB, Model, LinExpr, abs_, max_
from copy import copy
from tqdm import trange

from evanqp import Polytope
from evanqp.problems import CvxpyProblem, MPCProblem
from evanqp.layers import BoundArithmetic, InputLayer, ConcatLayer, LinearLayer, QPLayer, SeqLayer


class Verifier:

    def __init__(self, parameter_set, *problems):
        self.input_layer = InputLayer(parameter_set)

        self.problems = []
        for problem in problems:
            if isinstance(problem, CvxpyProblem):
                self.problems.append(QPLayer(problem, 1))
            else:
                self.problems.append(SeqLayer.from_pytorch(problem))

        self.bounds_calculated = False

    def compute_bounds(self, method=BoundArithmetic.ZONO_ARITHMETIC, **kwargs):
        self.input_layer.compute_bounds(method, **kwargs)
        for p in self.problems:
            p.compute_bounds(method, self.input_layer, **kwargs)
        self.bounds_calculated = True

        return [p.bounds['out'] for p in self.problems]

    def compute_ideal_cuts(self, model):
        ineqs = []
        for p in self.problems:
            ineqs += p.compute_ideal_cuts(model, self.input_layer, None)
        return ineqs

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

    def ideal_cuts_callback(self):
        def _callback(model, where):
            if where == GRB.Callback.MIPNODE:
                if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.Status.OPTIMAL:
                    # decrease cut freq with number of nodes explored
                    freq = model.cbGet(GRB.Callback.MIPNODE_NODCNT) + 1
                    if np.random.randint(0, freq, 1) == 0:
                        ineqs = self.compute_ideal_cuts(model)
                        for (lhs, rhs) in ineqs:
                            model.cbCut(lhs <= rhs)
        return _callback

    def warm_start(self, guess=None):
        if guess is None:
            guess = self.input_layer.forward(None)

        for problem in self.problems:
            problem.forward(guess, warm_start=True)

    def find_max_abs_diff(self, threads=0, output_flag=1, ideal_cuts=False, warm_start=True, guess=None):
        model = Model()
        model.setParam('OutputFlag', output_flag)
        model.setParam('Threads', threads)

        if len(self.problems) != 2:
            raise Exception('Number of problems must be 2.')
        if self.problems[0].out_size != self.problems[0].out_size:
            raise Exception('Problems do not have the same output size')

        self.setup_milp(model)
        if warm_start:
            self.warm_start(guess)

        diff = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY) for _ in range(self.problems[0].out_size)]
        abs_diff = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY) for _ in range(self.problems[0].out_size)]
        for i in range(self.problems[0].out_size):
            model.addConstr(diff[i] == self.problems[0].vars['out'][i] - self.problems[1].vars['out'][i])
            model.addConstr(abs_diff[i] == abs_(diff[i]))
        max_abs_diff = model.addVar(vtype=GRB.CONTINUOUS)
        model.addConstr(max_abs_diff == max_(abs_diff))

        model.setObjective(max_abs_diff, GRB.MAXIMIZE)
        model.update()
        if ideal_cuts:
            model.optimize(self.ideal_cuts_callback())
        else:
            model.optimize()

        return model.objBound, [p.x for p in self.input_layer.vars['out']]

    def verify_stability_sufficient(self, threads=0, output_flag=1, ideal_cuts=False, warm_start=True, guess=None):
        if len(self.problems) != 2:
            raise Exception('Number of problems must be 2.')
        if not isinstance(self.problems[0].problem, MPCProblem):
            raise Exception('The first problem must be of type MPCProblem.')

        mpc_problem = self.problems[0].problem

        model = Model()
        model.setParam('OutputFlag', output_flag)
        model.setParam('Threads', threads)

        if not self.bounds_calculated:
            self.compute_bounds()

        reduced_objective_problem = copy(mpc_problem)

        # monkey patch original mpc problem with reduced objective function
        def problem_patch(_self):
            return cp.Problem(_self.reduced_objective(), _self.original_problem().constraints)
        reduced_objective_problem.original_problem = reduced_objective_problem.problem
        reduced_objective_problem.problem = types.MethodType(problem_patch, reduced_objective_problem)

        reduced_objective_mpc_layer = QPLayer(reduced_objective_problem, 1)
        reduced_objective_mpc_layer.compute_bounds(BoundArithmetic.INT_ARITHMETIC, self.input_layer)

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

        if warm_start:
            self.warm_start(guess)

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

        # allow non-convex MIQP formulation
        if len(P_t_col_coef) > 0:
            model.setParam('NonConvex', 2)

        model.setObjective(obj, GRB.MINIMIZE)
        model.update()
        if ideal_cuts:
            model.optimize(self.ideal_cuts_callback())
        else:
            model.optimize()

        return model.objBound, [p.x for p in self.input_layer.vars['out']]

    def verify_stability_direct(self, threads=0, output_flag=1, ideal_cuts=False, warm_start=True, guess=None):
        if len(self.problems) != 2:
            raise Exception('Number of problems must be 2.')
        if not isinstance(self.problems[0].problem, MPCProblem):
            raise Exception('The first problem must be of type MPCProblem.')

        mpc_problem = self.problems[0].problem

        model = Model()
        model.setParam('OutputFlag', output_flag)
        model.setParam('Threads', threads)

        if not self.bounds_calculated:
            self.compute_bounds()

        x_u_layer = ConcatLayer(self.input_layer.out_size + self.problems[1].out_size, self.problems[1].depth + 1)
        A, B = self.problems[0].problem.dynamics()
        weight = np.hstack((A, B))
        bias = np.zeros(weight.shape[0])
        next_state_layer = LinearLayer(weight, bias, x_u_layer.depth + 1)
        mpc_cost_next_state_layer = QPLayer(mpc_problem, next_state_layer.depth + 1)

        x_u_layer.compute_bounds(BoundArithmetic.ZONO_ARITHMETIC, (self.input_layer, self.problems[1]))
        next_state_layer.compute_bounds(BoundArithmetic.ZONO_ARITHMETIC, x_u_layer)
        mpc_cost_next_state_layer.compute_bounds(BoundArithmetic.ZONO_ARITHMETIC, next_state_layer)

        self.input_layer.add_vars(model)
        self.problems[0].add_vars(model, only_primal=True)
        self.problems[1].add_vars(model)
        x_u_layer.add_vars(model)
        next_state_layer.add_vars(model)
        mpc_cost_next_state_layer.add_vars(model)
        model.update()

        self.input_layer.add_constr(model)
        self.problems[0].add_constr(model, self.input_layer, only_primal=True)
        self.problems[1].add_constr(model, self.input_layer)
        x_u_layer.add_constr(model, (self.input_layer, self.problems[1]))
        next_state_layer.add_constr(model, x_u_layer)
        mpc_cost_next_state_layer.add_constr(model, next_state_layer)
        model.update()

        if warm_start:
            self.warm_start(guess)

        x = self.problems[0].vars['x']
        x_t = mpc_cost_next_state_layer.vars['x']

        obj = 0
        P_row_idx, P_col_idx, P_col_coef = sp.find(self.problems[0].P)
        for i, j, Pij in zip(P_row_idx, P_col_idx, P_col_coef):
            obj += 0.5 * x[i] * Pij * x[j]
        obj += LinExpr(self.problems[0].q, x)
        P_t_row_idx, P_t_col_idx, P_t_col_coef = sp.find(mpc_cost_next_state_layer.P)
        for i, j, Pij in zip(P_t_row_idx, P_t_col_idx, P_t_col_coef):
            obj -= 0.5 * x_t[i] * Pij * x_t[j]
        obj -= LinExpr(mpc_cost_next_state_layer.q, x_t)

        # allow non-convex MIQP formulation
        if len(P_t_col_coef) > 0:
            model.setParam('NonConvex', 2)

        model.setObjective(obj, GRB.MINIMIZE)
        model.update()
        if ideal_cuts:
            model.optimize(self.ideal_cuts_callback())
        else:
            model.optimize()

        return model.objBound, [p.x for p in self.input_layer.vars['out']]

    def stable_region_outer_approx(self, seed_polytope, threads=0, output_flag=0, ideal_cuts=False, warm_start=True, guess=None):
        if len(self.problems) != 2:
            raise Exception('Number of problems must be 2.')
        if not isinstance(self.problems[0].problem, MPCProblem):
            raise Exception('The first problem must be of type MPCProblem.')

        mpc_problem = self.problems[0].problem

        model = Model()
        model.setParam('OutputFlag', output_flag)
        model.setParam('Threads', threads)

        if not self.bounds_calculated:
            self.compute_bounds()

        x_u_layer = ConcatLayer(self.input_layer.out_size + self.problems[1].out_size, self.problems[1].depth + 1)
        A, B = self.problems[0].problem.dynamics()
        weight = np.hstack((A, B))
        bias = np.zeros(weight.shape[0])
        next_state_layer = LinearLayer(weight, bias, x_u_layer.depth + 1)
        mpc_cost_next_state_layer = QPLayer(mpc_problem, next_state_layer.depth + 1)

        x_u_layer.compute_bounds(BoundArithmetic.ZONO_ARITHMETIC, (self.input_layer, self.problems[1]))
        next_state_layer.compute_bounds(BoundArithmetic.ZONO_ARITHMETIC, x_u_layer)
        mpc_cost_next_state_layer.compute_bounds(BoundArithmetic.ZONO_ARITHMETIC, next_state_layer)

        self.input_layer.add_vars(model)
        self.problems[0].add_vars(model, only_primal=True)
        self.problems[1].add_vars(model)
        x_u_layer.add_vars(model)
        next_state_layer.add_vars(model)
        mpc_cost_next_state_layer.add_vars(model)
        model.update()

        self.input_layer.add_constr(model)
        self.problems[0].add_constr(model, self.input_layer, only_primal=True)
        self.problems[1].add_constr(model, self.input_layer)
        x_u_layer.add_constr(model, (self.input_layer, self.problems[1]))
        next_state_layer.add_constr(model, x_u_layer)
        mpc_cost_next_state_layer.add_constr(model, next_state_layer)
        model.update()

        if warm_start:
            self.warm_start(guess)


        b = np.zeros(seed_polytope.b.shape)

        for i in trange(seed_polytope.A.shape[0]):
            model.setObjective(LinExpr(seed_polytope.A[i, :], self.input_layer.vars['out']), GRB.MAXIMIZE)
            model.update()
            if ideal_cuts:
                model.optimize(self.ideal_cuts_callback())
            else:
                model.optimize()

            b[i] = model.objVal

        return Polytope(seed_polytope.A, b)

    def variables_in_polytope(self, poly, eps=1e-6, threads=0, output_flag=1, warm_start=True, guess=None):
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
        if warm_start:
            self.warm_start(guess)

        for i in range(self.problems[0].out_size):
            model.setObjective(LinExpr(poly.A[i, :], self.problems[0].vars['out']) - poly.b[i], GRB.MAXIMIZE)
            model.update()
            model.optimize()

            if model.objBound > eps:
                return False

        return True

    @staticmethod
    def min_optimal_mpc_horizon(parameter_set, mpc_factory, poly, eps=1e-6, threads=0, warm_start=True, guess=None):
        N = 1
        mpc_problem = mpc_factory(N)
        print(f'Checking N = {N}')
        verifier = Verifier(parameter_set, mpc_problem)
        res = verifier.variables_in_polytope(poly, eps=eps, threads=threads, output_flag=0, warm_start=warm_start, guess=guess)
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
