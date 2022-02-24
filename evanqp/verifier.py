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

        self.model = None
        self.bounds_calculated = False
        self.bounds_lipschitz_calculated = False

    def compute_bounds(self, method=BoundArithmetic.ZONO_ARITHMETIC, **kwargs):
        self.input_layer.compute_bounds(method, **kwargs)
        for p in self.problems:
            p.compute_bounds(method, self.input_layer, **kwargs)
        self.bounds_calculated = True

        return [p.bounds['out'] for p in self.problems]

    def compute_bounds_lipschitz(self, **kwargs):
        if not self.bounds_calculated:
            self.compute_bounds(**kwargs)

        self.input_layer.compute_bounds_jacobian(**kwargs)
        for p in self.problems:
            p.compute_bounds_jacobian(self.input_layer, **kwargs)
        self.bounds_lipschitz_calculated = True

        return [p.bounds['out'] for p in self.problems]

    def compute_ideal_cuts(self, model):
        ineqs = []
        for p in self.problems:
            ineqs += p.compute_ideal_cuts(model, self.input_layer, None)
        return ineqs

    def setup_milp(self):
        if not self.bounds_calculated:
            self.compute_bounds()

        self.input_layer.add_vars(self.model)
        for p in self.problems:
            p.add_vars(self.model)
        self.model.update()

        self.input_layer.add_constr(self.model)
        for p in self.problems:
            p.add_constr(self.model, self.input_layer)
        self.model.update()

    def setup_milp_lipschitz(self):
        if not self.bounds_lipschitz_calculated:
            self.compute_bounds_lipschitz()

        self.input_layer.add_vars_jacobian(self.model, None)
        for p in self.problems:
            p.add_vars_jacobian(self.model, self.input_layer)
        self.model.update()

        self.input_layer.add_constr_jacobian(self.model, None)
        for p in self.problems:
            p.add_constr_jacobian(self.model, self.input_layer)
        self.model.update()

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

    def approximation_error(self, norm=np.inf, threads=0, output_flag=1, feasibility_tol=None, int_feas_tol=None, optimality_tol=None, ideal_cuts=False, warm_start=True, guess=None):
        if len(self.problems) != 2:
            raise Exception('Number of problems must be 2.')
        if self.problems[0].out_size != self.problems[0].out_size:
            raise Exception('Problems do not have the same output size')

        self.model = Model()
        self.model.setParam('OutputFlag', output_flag)
        self.model.setParam('Threads', threads)
        if feasibility_tol is not None:
            self.model.setParam('FeasibilityTol', feasibility_tol)
        if int_feas_tol is not None:
            self.model.setParam('IntFeasTol', int_feas_tol)
        if optimality_tol is not None:
            self.model.setParam('OptimalityTol', optimality_tol)

        self.setup_milp()
        if warm_start:
            self.warm_start(guess)

        diff = [self.model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY) for _ in range(self.problems[0].out_size)]
        abs_diff = [self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY) for _ in range(self.problems[0].out_size)]
        for i in range(self.problems[0].out_size):
            self.model.addConstr(diff[i] == self.problems[0].vars['out'][i] - self.problems[1].vars['out'][i])
            self.model.addConstr(abs_diff[i] == abs_(diff[i]))

        error_norm = self.model.addVar(vtype=GRB.CONTINUOUS)
        if norm == np.inf:
            self.model.addConstr(error_norm == max_(abs_diff))
        elif norm == 1:
            self.model.addConstr(error_norm == sum(abs_diff))
        else:
            raise Exception('Norm can only be inf(np.inf) or 1')

        self.model.setObjective(error_norm, GRB.MAXIMIZE)
        self.model.update()
        if ideal_cuts:
            self.model.optimize(self.ideal_cuts_callback())
        else:
            self.model.optimize()

        return self.model.objBound, np.array([p.x for p in self.input_layer.vars['out']])

    def verify_stability_sufficient(self, threads=0, output_flag=1, feasibility_tol=None, int_feas_tol=None, optimality_tol=None, ideal_cuts=False, warm_start=True, guess=None):
        if len(self.problems) != 2:
            raise Exception('Number of problems must be 2.')
        if not isinstance(self.problems[0].problem, MPCProblem):
            raise Exception('The first problem must be of type MPCProblem.')

        mpc_problem = self.problems[0].problem

        self.model = Model()
        self.model.setParam('OutputFlag', output_flag)
        self.model.setParam('Threads', threads)
        if feasibility_tol is not None:
            self.model.setParam('FeasibilityTol', feasibility_tol)
        if int_feas_tol is not None:
            self.model.setParam('IntFeasTol', int_feas_tol)
        if optimality_tol is not None:
            self.model.setParam('OptimalityTol', optimality_tol)

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

        self.input_layer.add_vars(self.model)
        self.problems[0].add_vars(self.model, only_primal=True)
        reduced_objective_mpc_layer.add_vars(self.model)
        self.problems[1].add_vars(self.model)
        self.model.update()

        self.input_layer.add_constr(self.model)
        self.problems[0].add_constr(self.model, self.input_layer, only_primal=True)
        reduced_objective_mpc_layer.add_constr(self.model, self.input_layer)
        self.problems[1].add_constr(self.model, self.input_layer)
        self.model.update()

        for i in range(reduced_objective_mpc_layer.out_size):
            self.model.addConstr(reduced_objective_mpc_layer.vars['out'][i] == self.problems[1].vars['out'][i])
        self.model.update()

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
            self.model.setParam('NonConvex', 2)

        self.model.setObjective(obj, GRB.MINIMIZE)
        self.model.update()
        if ideal_cuts:
            self.model.optimize(self.ideal_cuts_callback())
        else:
            self.model.optimize()

        return self.model.objBound, np.array([p.x for p in self.input_layer.vars['out']])

    def verify_stability_direct(self, threads=0, output_flag=1, feasibility_tol=None, int_feas_tol=None, optimality_tol=None, ideal_cuts=False, warm_start=True, guess=None):
        if len(self.problems) != 2:
            raise Exception('Number of problems must be 2.')
        if not isinstance(self.problems[0].problem, MPCProblem):
            raise Exception('The first problem must be of type MPCProblem.')

        mpc_problem = self.problems[0].problem

        self.model = Model()
        self.model.setParam('OutputFlag', output_flag)
        self.model.setParam('Threads', threads)
        if feasibility_tol is not None:
            self.model.setParam('FeasibilityTol', feasibility_tol)
        if int_feas_tol is not None:
            self.model.setParam('IntFeasTol', int_feas_tol)
        if optimality_tol is not None:
            self.model.setParam('OptimalityTol', optimality_tol)

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

        self.input_layer.add_vars(self.model)
        self.problems[0].add_vars(self.model, only_primal=True)
        self.problems[1].add_vars(self.model)
        x_u_layer.add_vars(self.model)
        next_state_layer.add_vars(self.model)
        mpc_cost_next_state_layer.add_vars(self.model)
        self.model.update()

        self.input_layer.add_constr(self.model)
        self.problems[0].add_constr(self.model, self.input_layer, only_primal=True)
        self.problems[1].add_constr(self.model, self.input_layer)
        x_u_layer.add_constr(self.model, (self.input_layer, self.problems[1]))
        next_state_layer.add_constr(self.model, x_u_layer)
        mpc_cost_next_state_layer.add_constr(self.model, next_state_layer)
        self.model.update()

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
            self.model.setParam('NonConvex', 2)

        self.model.setObjective(obj, GRB.MINIMIZE)
        self.model.update()
        if ideal_cuts:
            self.model.optimize(self.ideal_cuts_callback())
        else:
            self.model.optimize()

        return self.model.objBound, np.array([p.x for p in self.input_layer.vars['out']])

    def stable_region_outer_approx(self, seed_polytope, threads=0, output_flag=0, feasibility_tol=None, int_feas_tol=None, optimality_tol=None, ideal_cuts=False, warm_start=True, guess=None):
        if len(self.problems) != 2:
            raise Exception('Number of problems must be 2.')
        if not isinstance(self.problems[0].problem, MPCProblem):
            raise Exception('The first problem must be of type MPCProblem.')

        mpc_problem = self.problems[0].problem

        self.model = Model()
        self.model.setParam('OutputFlag', output_flag)
        self.model.setParam('Threads', threads)
        if feasibility_tol is not None:
            self.model.setParam('FeasibilityTol', feasibility_tol)
        if int_feas_tol is not None:
            self.model.setParam('IntFeasTol', int_feas_tol)
        if optimality_tol is not None:
            self.model.setParam('OptimalityTol', optimality_tol)

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

        self.input_layer.add_vars(self.model)
        self.problems[0].add_vars(self.model, only_primal=True)
        self.problems[1].add_vars(self.model)
        x_u_layer.add_vars(self.model)
        next_state_layer.add_vars(self.model)
        mpc_cost_next_state_layer.add_vars(self.model)
        self.model.update()

        self.input_layer.add_constr(self.model)
        self.problems[0].add_constr(self.model, self.input_layer, only_primal=True)
        self.problems[1].add_constr(self.model, self.input_layer)
        x_u_layer.add_constr(self.model, (self.input_layer, self.problems[1]))
        next_state_layer.add_constr(self.model, x_u_layer)
        mpc_cost_next_state_layer.add_constr(self.model, next_state_layer)
        self.model.update()

        if warm_start:
            self.warm_start(guess)


        b = np.zeros(seed_polytope.b.shape)

        for i in trange(seed_polytope.A.shape[0]):
            self.model.setObjective(LinExpr(seed_polytope.A[i, :], self.input_layer.vars['out']), GRB.MAXIMIZE)
            self.model.update()
            if ideal_cuts:
                self.model.optimize(self.ideal_cuts_callback())
            else:
                self.model.optimize()

            b[i] = self.model.objVal

        return Polytope(seed_polytope.A, b)

    def lipschitz_constant(self, norm=np.inf, threads=0, output_flag=1, feasibility_tol=None, int_feas_tol=None, optimality_tol=None, ideal_cuts=False, warm_start=False, guess=None):
        if len(self.problems) != 1:
            raise Exception('Number of problems must be 1.')

        self.model = Model()
        self.model.setParam('OutputFlag', output_flag)
        self.model.setParam('Threads', threads)
        if feasibility_tol is not None:
            self.model.setParam('FeasibilityTol', feasibility_tol)
        if int_feas_tol is not None:
            self.model.setParam('IntFeasTol', int_feas_tol)
        if optimality_tol is not None:
            self.model.setParam('OptimalityTol', optimality_tol)

        self.setup_milp()
        self.setup_milp_lipschitz()

        if warm_start:
            self.warm_start(guess)

        abs_jac_T = np.empty((self.problems[0].vars['out_jac'].shape[1], self.problems[0].vars['out_jac'].shape[0]), dtype=object)
        for i in range(self.problems[0].vars['out_jac'].shape[0]):
            for j in range(self.problems[0].vars['out_jac'].shape[1]):
                abs_jac_T[j, i] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
                self.model.addConstr(abs_jac_T[j, i] == abs_(self.problems[0].vars['out_jac'][i, j]))

        lipschitz_norm = self.model.addVar(vtype=GRB.CONTINUOUS)
        if norm == np.inf:
            lipschitz_norm_sum = [self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY) for _ in range(self.problems[0].vars['out_jac'].shape[1])]
            for i in range(self.problems[0].vars['out_jac'].shape[1]):
                self.model.addConstr(lipschitz_norm_sum[i] == sum(abs_jac_T[i, :]))
            self.model.addConstr(lipschitz_norm == max_(lipschitz_norm_sum))
        elif norm == 1:
            lipschitz_norm_sum = [self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY) for _ in range(self.problems[0].vars['out_jac'].shape[0])]
            for j in range(self.problems[0].vars['out_jac'].shape[0]):
                self.model.addConstr(lipschitz_norm_sum[j] == sum(abs_jac_T[:, j]))
            self.model.addConstr(lipschitz_norm == max_(lipschitz_norm_sum))
        else:
            raise Exception('Norm can only be inf(np.inf) or 1')

        self.model.setObjective(lipschitz_norm, GRB.MAXIMIZE)
        self.model.update()
        if ideal_cuts:
            self.model.optimize(self.ideal_cuts_callback())
        else:
            self.model.optimize()

        jacobian = np.zeros(self.problems[0].vars['out_jac'].shape)
        for i in range(self.problems[0].vars['out_jac'].shape[0]):
            for j in range(self.problems[0].vars['out_jac'].shape[1]):
                jacobian[i, j] = self.problems[0].vars['out_jac'][i, j].x

        return self.model.objBound, jacobian, np.array([p.x for p in self.input_layer.vars['out']])

    def approximation_error_lipschitz_constant(self, norm=np.inf, threads=0, output_flag=1, feasibility_tol=None, int_feas_tol=None, optimality_tol=None, ideal_cuts=False, warm_start=False, guess=None):
        if len(self.problems) != 2:
            raise Exception('Number of problems must be 2.')
        if self.problems[0].out_size != self.problems[0].out_size:
            raise Exception('Problems do not have the same output size')

        self.model = Model()
        self.model.setParam('OutputFlag', output_flag)
        self.model.setParam('Threads', threads)
        if feasibility_tol is not None:
            self.model.setParam('FeasibilityTol', feasibility_tol)
        if int_feas_tol is not None:
            self.model.setParam('IntFeasTol', int_feas_tol)
        if optimality_tol is not None:
            self.model.setParam('OptimalityTol', optimality_tol)

        self.setup_milp()
        self.setup_milp_lipschitz()

        if warm_start:
            self.warm_start(guess)

        diff_jac_T = np.empty((self.problems[0].vars['out_jac'].shape[1], self.problems[0].vars['out_jac'].shape[0]), dtype=object)
        abs_diff_jac_T = np.empty((self.problems[0].vars['out_jac'].shape[1], self.problems[0].vars['out_jac'].shape[0]), dtype=object)
        for i in range(self.problems[0].vars['out_jac'].shape[0]):
            for j in range(self.problems[0].vars['out_jac'].shape[1]):
                diff_jac_T[j, i] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
                abs_diff_jac_T[j, i] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY)
                self.model.addConstr(diff_jac_T[j, i] == self.problems[0].vars['out_jac'][i, j] - self.problems[1].vars['out_jac'][i, j])
                self.model.addConstr(abs_diff_jac_T[j, i] == abs_(diff_jac_T[j, i]))

        lipschitz_norm = self.model.addVar(vtype=GRB.CONTINUOUS)
        if norm == np.inf:
            lipschitz_norm_sum = [self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY) for _ in range(self.problems[0].vars['out_jac'].shape[1])]
            for i in range(self.problems[0].vars['out_jac'].shape[1]):
                self.model.addConstr(lipschitz_norm_sum[i] == sum(abs_diff_jac_T[i, :]))
            self.model.addConstr(lipschitz_norm == max_(lipschitz_norm_sum))
        elif norm == 1:
            lipschitz_norm_sum = [self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY) for _ in
                                  range(self.problems[0].vars['out_jac'].shape[0])]
            for j in range(self.problems[0].vars['out_jac'].shape[0]):
                self.model.addConstr(lipschitz_norm_sum[j] == sum(abs_diff_jac_T[:, j]))
            self.model.addConstr(lipschitz_norm == max_(lipschitz_norm_sum))
        else:
            raise Exception('Norm can only be inf(np.inf) or 1')

        self.model.setObjective(lipschitz_norm, GRB.MAXIMIZE)
        self.model.update()
        if ideal_cuts:
            self.model.optimize(self.ideal_cuts_callback())
        else:
            self.model.optimize()

        jacobian0 = np.zeros(self.problems[0].vars['out_jac'].shape)
        for i in range(self.problems[0].vars['out_jac'].shape[0]):
            for j in range(self.problems[0].vars['out_jac'].shape[1]):
                jacobian0[i, j] = self.problems[0].vars['out_jac'][i, j].x

        jacobian1 = np.zeros(self.problems[1].vars['out_jac'].shape)
        for i in range(self.problems[1].vars['out_jac'].shape[0]):
            for j in range(self.problems[1].vars['out_jac'].shape[1]):
                jacobian1[i, j] = self.problems[1].vars['out_jac'][i, j].x

        return self.model.objBound, jacobian0, jacobian1, np.array([p.x for p in self.input_layer.vars['out']])

    def variables_in_polytope(self, poly, eps=1e-6, threads=0, output_flag=1, warm_start=True, guess=None):
        if len(self.problems) != 1:
            raise Exception('Number of problems must be 1.')
        if not isinstance(poly, Polytope):
            raise Exception('poly must be of type Polytope.')
        if self.problems[0].out_size != poly.A.shape[1]:
            raise Exception('poly shape does not match problem output size.')

        self.model = Model()
        self.model.setParam('OutputFlag', output_flag)
        self.model.setParam('Threads', threads)

        self.setup_milp()
        if warm_start:
            self.warm_start(guess)

        for i in range(self.problems[0].out_size):
            self.model.setObjective(LinExpr(poly.A[i, :], self.problems[0].vars['out']) - poly.b[i], GRB.MAXIMIZE)
            self.model.update()
            self.model.optimize()

            if self.model.objBound > eps:
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
