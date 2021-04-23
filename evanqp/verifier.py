import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from gurobipy import GRB, Model, LinExpr, abs_, max_
from tqdm import tqdm

from evanqp import SeqNet, QPProblem, MPCProblem
from evanqp.sets import Set, Box, Polytope


class NNProcessor:

    def __init__(self, net, parameter_set):
        self.net = net
        self.n_layers = len(net.blocks)

        if not isinstance(parameter_set, Set):
            raise Exception('Parameter set is required for Neural Networks.')
        self.parameter_set = parameter_set

        self.active_relus = None
        self.inactive_relus = None
        self.total_relus = None
        self.active_relu_count = None
        self.inactive_relu_count = None

        self.lbs = {}
        self.ubs = {}

        if isinstance(self.net.blocks[0], nn.Linear):
            self.parameter_size = self.net.blocks[0].weight.detach().cpu().size()[1]
        else:
            raise NotImplementedError()

        if isinstance(self.net.blocks[-1], nn.Linear):
            self.variable_size = self.net.blocks[-1].weight.detach().cpu().size()[0]
        else:
            raise NotImplementedError()

    def optimize_relus(self, dataset, verbose=False):
        active_relus = {}
        inactive_relus = {}

        device = next(self.net.parameters()).device
        dataloader = DataLoader(dataset, batch_size=256)

        self.net.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(dataloader):
                data = data.to(device)
                for lidx in range(self.n_layers):
                    if isinstance(self.net.blocks[lidx], nn.ReLU):
                        x = self.net.forward_until(lidx, data)
                        x_min, _ = torch.min(x, dim=0)
                        x_max, _ = torch.max(x, dim=0)
                        if lidx in active_relus:
                            active_relus[lidx] = torch.bitwise_and(active_relus[lidx], x_min.detach().cpu() > 0)
                            inactive_relus[lidx] = torch.bitwise_and(inactive_relus[lidx], x_max.detach().cpu() == 0)
                        else:
                            active_relus[lidx] = x_min.detach().cpu() > 0
                            inactive_relus[lidx] = x_max.detach().cpu() == 0

        self.active_relus = active_relus
        self.inactive_relus = inactive_relus

        self.total_relus = 0
        self.active_relu_count = 0
        self.inactive_relu_count = 0
        for lidx in active_relus.keys():
            self.total_relus += active_relus[lidx].size()[0]
            self.active_relu_count += torch.sum(active_relus[lidx]).item()
            self.inactive_relu_count += torch.sum(inactive_relus[lidx]).item()

        if verbose:
            print(f'Active ReLU neurons: {self.active_relu_count}/{self.total_relus}')
            print(f'Inactive ReLU neurons: {self.inactive_relu_count}/{self.total_relus}')

    def construct_gurobi_model(self, model=None, neurons=None, milp_tightening=False, lp_relaxation=False, guess=None):
        if isinstance(self.parameter_set, Box):
            lb = torch.from_numpy(self.parameter_set.lb).float()
            ub = torch.from_numpy(self.parameter_set.ub).float()
        elif isinstance(self.parameter_set, Polytope):
            lb = torch.from_numpy(self.parameter_set.bounding_box().lb).float()
            ub = torch.from_numpy(self.parameter_set.bounding_box().ub).float()
        else:
            raise NotImplementedError()

        if model is not None:
            neurons[-1] = [model.addVar(vtype=GRB.CONTINUOUS, lb=lb.numpy()[i], ub=ub.numpy()[i], name=f'in_{i}') for i in range(self.parameter_size)]
            if isinstance(self.parameter_set, Polytope):
                A, b = self.parameter_set.A, self.parameter_set.b
                for i in range(A.shape[0]):
                    model.addConstr(LinExpr(A[i, :], neurons[-1]) <= b[i])
            model.update()

        self.net.eval()
        with torch.no_grad():
            for lidx in tqdm(range(self.n_layers)):
                if isinstance(self.net.blocks[lidx], nn.Linear):
                    weight = self.net.blocks[lidx].weight.detach().cpu()
                    bias = self.net.blocks[lidx].bias.detach().cpu()
                    n_outs = weight.size()[0]

                    lb_new = torch.clamp(weight, max=0) @ ub + torch.clamp(weight, min=0) @ lb + bias
                    ub_new = torch.clamp(weight, min=0) @ ub + torch.clamp(weight, max=0) @ lb + bias

                    if model is not None:
                        neurons[lidx] = []
                        for i in range(n_outs):
                            neurons[lidx] += [model.addVar(vtype=GRB.CONTINUOUS, lb=lb_new[i].item(), ub=ub_new[i].item(), name=f'n_{lidx}_{i}')]
                            model.addConstr(neurons[lidx][i] == LinExpr(weight[i, :].detach().cpu().numpy(), neurons[lidx - 1]) + bias[i].item())
                            model.update()

                    if model is not None and milp_tightening:
                        for i in range(n_outs):
                            model.setObjective(neurons[lidx][i], GRB.MINIMIZE)
                            model.update()
                            model.optimize()
                            lb_new[i] = max(model.objBound, lb_new[i])

                            model.setObjective(neurons[lidx][i], GRB.MAXIMIZE)
                            model.update()
                            model.optimize()
                            ub_new[i] = min(model.objBound, ub_new[i])

                    lb = lb_new
                    ub = ub_new

                elif isinstance(self.net.blocks[lidx], nn.ReLU):
                    if lidx in self.lbs:
                        # load already tightened bounds
                        lb = torch.max(lb, self.lbs[lidx])
                        ub = torch.min(ub, self.ubs[lidx])

                    # save tightened bounds for ReLU inputs
                    self.lbs[lidx] = lb
                    self.ubs[lidx] = ub

                    if model is not None:
                        if guess is not None:
                            x_prev = self.net.forward_until(lidx - 1, torch.from_numpy(guess).float())
                            x = torch.relu(x_prev)
                            if self.active_relus is not None:
                                x[self.active_relus[lidx]] = x_prev[self.active_relus[lidx]]
                                x *= torch.bitwise_not(self.inactive_relus[lidx])

                        neurons[lidx] = []
                        for i in range(len(neurons[lidx - 1])):
                            if (self.active_relus is not None and self.active_relus[lidx][i]) or lb[i] >= 0:
                                neurons[lidx] += [neurons[lidx - 1][i]]
                            elif (self.inactive_relus is not None and self.inactive_relus[lidx][i]) or ub[i] <= 0:
                                neurons[lidx] += [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=0, name=f'n_{lidx}_{i}')]
                            else:
                                neurons[lidx] += [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=ub[i].item(), name=f'n_{lidx}_{i}')]
                                if lp_relaxation:
                                    relu_ind = model.addVar(vtype=GRB.CONTINUOUS, name=f'relu_ind{lidx}_{i}')
                                else:
                                    relu_ind = model.addVar(vtype=GRB.BINARY, name=f'relu_ind{lidx}_{i}')
                                model.addConstr(neurons[lidx][i] >= 0)
                                model.addConstr(neurons[lidx][i] >= neurons[lidx - 1][i])
                                model.addConstr(neurons[lidx][i] <= ub[i].item() * relu_ind)
                                model.addConstr(neurons[lidx][i] <= neurons[lidx - 1][i] - lb[i].item() * (1 - relu_ind))
                                model.addConstr((relu_ind == 1) >> (neurons[lidx - 1][i] >= 0))
                                model.addConstr((relu_ind == 0) >> (neurons[lidx - 1][i] <= 0))

                                if guess is not None:
                                    if x[i].item() > 0:
                                        relu_ind.start = 1
                                    else:
                                        relu_ind.start = 0

                            model.update()

                    lb = torch.clamp(lb, min=0)
                    ub = torch.clamp(ub, min=0)
                    if self.inactive_relus is not None:
                        lb *= torch.bitwise_not(self.inactive_relus[lidx])
                        ub *= torch.bitwise_not(self.inactive_relus[lidx])
                else:
                    raise NotImplementedError()

        return lb, ub


def replace_params(expr, param_dic):
    for idx, arg in enumerate(expr.args):
        if isinstance(arg, cp.Parameter):
            param_dic = replace_param(expr, idx, param_dic)
        else:
            param_dic = replace_params(arg, param_dic)
    return param_dic


def replace_param(expr, idx, param_dic):
    param = expr.args[idx]
    placeholder = cp.Variable(param.shape, var_id=param.id, name=param.name())
    expr.args[idx] = placeholder
    param_dic[placeholder.id] = (expr, idx, param)
    return param_dic


def restore_params(expr, param_dic):
    for idx, arg in enumerate(expr.args):
        if isinstance(arg, cp.Variable) and arg.id in param_dic:
            expr.args[idx] = param_dic[arg.id][2]
        else:
            restore_params(arg, param_dic)


def compile_qp_problem_with_params_as_variables(problem):
    objective = problem.objective
    constraints = problem.constraints

    param_dic = {}
    replace_params(objective, param_dic)
    for con in constraints:
        replace_params(con, param_dic)
    problem = cp.Problem(objective, constraints)

    data, chain, inverse_data = problem.get_problem_data(cp.OSQP)
    compiler = data[cp.settings.PARAM_PROB]

    P = data['P']
    q = data['q']
    A = data['A']
    b = data['b']
    F = data['F']
    g = data['G']

    restore_params(objective, param_dic)
    for con in constraints:
        restore_params(con, param_dic)

    return P, q, A, b, F, g, compiler.var_id_to_col


class CompiledQPProblem:

    def __init__(self, problem):
        P, q, A, b, F, g, var_id_to_col = compile_qp_problem_with_params_as_variables(problem)
        self.P = P
        self.q = q
        self.A = A
        self.b = b
        self.b_guess = b
        self.F = F
        self.g = g
        self.var_id_to_col = var_id_to_col

    def add_eq_constraints(self, A, b, b_guess=None):
        self.A = sp.vstack((self.A, A), format='csc')
        self.b = np.concatenate((self.b, b))
        if self.b_guess is not None and b_guess is not None:
            self.b_guess = np.concatenate((self.b_guess, b_guess))
        else:
            self.b_guess = None

    def construct_gurobi_model(self, model, only_primal=False):
        if self.b_guess is not None:
            x = cp.Variable(self.P.shape[1])
            obj = cp.Minimize(0.5 * cp.quad_form(x, self.P) + self.q.T @ x)
            eq_con = self.A @ x == self.b_guess
            ineq_con = self.F @ x <= self.g
            prob = cp.Problem(obj, [eq_con, ineq_con])
            prob.solve(solver=cp.GUROBI)

        x = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'x_{i}') for i in range(self.P.shape[1])]
        if not only_primal:
            mu = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'mu_{i}') for i in range(self.A.shape[0])]
            lam = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name=f'lam_{i}') for i in range(self.F.shape[0])]
        model.update()

        if not only_primal:
            for i in range(self.P.shape[0]):
                _, P_col_idx, P_col_coef = sp.find(self.P[i, :])
                _, A_col_idx, A_col_coef = sp.find(self.A.T[i, :])
                _, F_col_idx, F_col_coef = sp.find(self.F.T[i, :])
                model.addConstr(LinExpr(P_col_coef, [x[j] for j in P_col_idx])
                                + self.q[i]
                                + LinExpr(A_col_coef, [mu[j] for j in A_col_idx])
                                + LinExpr(F_col_coef, [lam[j] for j in F_col_idx])
                                == 0)
        for i in range(self.A.shape[0]):
            _, A_col_idx, A_col_coef = sp.find(self.A[i, :])
            model.addConstr(LinExpr(A_col_coef, [x[j] for j in A_col_idx]) - self.b[i] == 0)
        for i in range(self.F.shape[0]):
            _, F_col_idx, F_col_coef = sp.find(self.F[i, :])
            model.addConstr(LinExpr(F_col_coef, [x[j] for j in F_col_idx]) - self.g[i] <= 0)
        model.update()

        if not only_primal:
            r = [model.addVar(vtype=GRB.BINARY, name=f'r_{i}') for i in range(self.F.shape[0])]
            model.update()
            for i in range(self.F.shape[0]):
                # M = 1e5
                # model.addConstr(F[i, :] @ x - g[i] >= -r[i] * M)
                # model.addConstr(lam[i] <= (1 - r[i]) * M)
                _, F_col_idx, F_col_coef = sp.find(self.F[i, :])
                model.addConstr((r[i] == 0) >> (LinExpr(F_col_coef, [x[j] for j in F_col_idx]) - self.g[i] == 0))
                model.addConstr((r[i] == 1) >> (lam[i] == 0))

                if self.b_guess is not None and not prob.status in [cp.INFEASIBLE, cp.UNBOUNDED]:
                    if abs(ineq_con.dual_value[i]) >= 1e-10:
                        r[i].start = 0
                    else:
                        r[i].start = 1

            model.update()

        if not only_primal:
            return x, mu, lam, r
        return x


class Verifier:

    def __init__(self, ref_problem, approx_problem, parameter_set=None):
        if isinstance(ref_problem, SeqNet):
            self.ref_problem = NNProcessor(ref_problem, parameter_set)
        else:
            self.ref_problem = ref_problem

        if isinstance(approx_problem, SeqNet):
            self.approx_problem = NNProcessor(approx_problem, parameter_set)
        else:
            self.approx_problem = approx_problem

        self.parameter_set = parameter_set

    def optimize_relus(self, dataset, only_ref_problem=False, only_approx_problem=False, verbose=False):
        if isinstance(self.ref_problem, NNProcessor) and not only_approx_problem:
            if verbose:
                print('Reference Problem:')
            self.ref_problem.optimize_relus(dataset, verbose)

        if isinstance(self.approx_problem, NNProcessor) and not only_ref_problem:
            if verbose:
                print('Approximate Problem:')
            self.approx_problem.optimize_relus(dataset, verbose)

    def interval_tightening(self, only_ref_problem=False, only_approx_problem=False, verbose=False):
        if isinstance(self.ref_problem, NNProcessor) and not only_approx_problem:
            # constructing without model will only calculate interval bounds
            lb, ub = self.ref_problem.construct_gurobi_model()
            if verbose:
                print('Reference Problem Bounds for Variables:')
                print(f'lower bound: {lb}')
                print(f'upper bound: {ub}')

        if isinstance(self.approx_problem, NNProcessor) and not only_ref_problem:
            # constructing without model will only calculate interval bounds
            lb, ub = self.approx_problem.construct_gurobi_model()
            if verbose:
                print('Approximate Problem Bounds for Variables:')
                print(f'lower bound: {lb}')
                print(f'upper bound: {ub}')

    def lp_tightening(self, only_ref_problem=False, only_approx_problem=False, verbose=False):
        if isinstance(self.ref_problem, NNProcessor) and not only_approx_problem:
            model = Model('Reference Problem LP Tightening')
            model.setParam('OutputFlag', 0)
            lb, ub = self.ref_problem.construct_gurobi_model(
                model=model,
                neurons={},
                milp_tightening=True,
                lp_relaxation=True)
            if verbose:
                print('Reference Problem Bounds for Variables:')
                print(f'lower bound: {lb}')
                print(f'upper bound: {ub}')

        if isinstance(self.approx_problem, NNProcessor) and not only_ref_problem:
            model = Model('Approximate Problem LP Tightening')
            model.setParam('OutputFlag', 0)
            lb, ub = self.approx_problem.construct_gurobi_model(
                model=model,
                neurons={},
                milp_tightening=True,
                lp_relaxation=True)
            if verbose:
                print('Approximate Problem Bounds for Variables:')
                print(f'lower bound: {lb}')
                print(f'upper bound: {ub}')

    def milp_tightening(self, only_ref_problem=False, only_approx_problem=False, verbose=False):
        if isinstance(self.ref_problem, NNProcessor) and not only_approx_problem:
            model = Model('Reference Problem MILP Tightening')
            model.setParam('OutputFlag', 0)
            lb, ub = self.ref_problem.construct_gurobi_model(
                model=model,
                neurons={},
                milp_tightening=True,
                lp_relaxation=False)
            if verbose:
                print('Reference Problem Bounds for Variables:')
                print(f'lower bound: {lb}')
                print(f'upper bound: {ub}')

        if isinstance(self.approx_problem, NNProcessor) and not only_ref_problem:
            model = Model('Approximate Problem MILP Tightening')
            model.setParam('OutputFlag', 0)
            lb, ub = self.approx_problem.construct_gurobi_model(
                model=model,
                neurons={},
                milp_tightening=True,
                lp_relaxation=False)
            if verbose:
                print('Approximate Problem Bounds for Variables:')
                print(f'lower bound: {lb}')
                print(f'upper bound: {ub}')

    def find_max_abs_diff(self, guess=None, threads=0):
        model = Model('Max Abs Diff MILP')
        model.setParam('Threads', threads)

        if isinstance(self.ref_problem, NNProcessor):
            parameter_size = self.ref_problem.parameter_size
            variable_size = self.ref_problem.variable_size
        elif isinstance(self.ref_problem, QPProblem):
            parameter_size = self.ref_problem.parameter_size()
            variable_size = self.ref_problem.variable_size()

        parameters = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'p_{i}') for i in range(parameter_size)]
        ref_variables = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'rv_{i}') for i in range(variable_size)]
        approx_variables = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'av_{i}') for i in range(variable_size)]
        model.update()

        self._constrain_parameters(model, parameters)
        self._connect_problem(self.ref_problem, model, parameters, ref_variables, guess=guess)
        self._connect_problem(self.approx_problem, model, parameters, approx_variables, guess=guess)

        diff = [model.addVar(vtype=GRB.CONTINUOUS, name=f'diff_{i}') for i in range(variable_size)]
        abs_diff = [model.addVar(vtype=GRB.CONTINUOUS, name=f'abs_diff_{i}') for i in range(variable_size)]
        for i in range(variable_size):
            model.addConstr(diff[i] == ref_variables[i] - approx_variables[i])
            model.addConstr(abs_diff[i] == abs_(diff[i]))
        max_abs_diff = model.addVar(vtype=GRB.CONTINUOUS, name='max_abs_diff')
        model.addConstr(max_abs_diff == max_(abs_diff))

        model.setObjective(max_abs_diff, GRB.MAXIMIZE)
        model.update()
        model.optimize()

        return model.objBound, [p.x for p in parameters]

    def verify_stability(self, guess=None, threads=0):
        if not isinstance(self.ref_problem, MPCProblem):
            raise Exception('The reference problem must be of type MPCProblem.')

        parameter_size = self.ref_problem.parameter_size()
        variable_size = self.ref_problem.variable_size()

        model = Model('Max Abs Diff MILP')
        model.setParam('Threads', threads)
        model.setParam('NonConvex', 2)  # allow non-convex MIQP formulation

        parameters = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'p_{i}') for i in range(parameter_size)]
        approx_variables = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'av_{i}') for i in range(variable_size)]
        model.update()

        self._constrain_parameters(model, parameters)
        self._connect_problem(self.approx_problem, model, parameters, approx_variables, guess=guess)

        reduced_objective = self.ref_problem.reduced_objective()
        reduced_objective_problem = cp.Problem(reduced_objective, self.ref_problem.problem().constraints)

        compiled = CompiledQPProblem(self.ref_problem.problem())
        reduced_compiled = CompiledQPProblem(reduced_objective_problem)

        A_con = np.zeros((parameter_size + variable_size, reduced_compiled.A.shape[1]))
        b_con = np.zeros(parameter_size + variable_size, dtype=object)
        i = 0
        for j in range(len(self.ref_problem.parameters())):
            idx = reduced_compiled.var_id_to_col[self.ref_problem.parameters()[j].id]
            size = self.ref_problem.parameters()[j].size
            for k in range(size):
                A_con[i, idx + k] = 1
                b_con[i] = parameters[i]
                i += 1
        i = 0
        for j in range(len(self.ref_problem.variables())):
            idx = reduced_compiled.var_id_to_col[self.ref_problem.variables()[j].id]
            size = self.ref_problem.variables()[j].size
            for k in range(size):
                A_con[parameter_size + i, idx + k] = 1
                b_con[parameter_size + i] = approx_variables[i]
                i += 1
        reduced_compiled.add_eq_constraints(A_con, b_con)

        A_par = np.zeros((parameter_size, compiled.A.shape[1]))
        b_par = np.zeros(parameter_size, dtype=object)
        if guess is None:
            b_guess = None
        else:
            b_guess = np.zeros(parameter_size)
        i = 0
        for j in range(len(self.ref_problem.parameters())):
            idx = compiled.var_id_to_col[self.ref_problem.parameters()[j].id]
            size = self.ref_problem.parameters()[j].size
            for k in range(size):
                A_par[i, idx + k] = 1
                b_par[i] = parameters[i]
                if guess is not None:
                    b_guess[i] = guess[i]
                i += 1
        compiled.add_eq_constraints(A_par, b_par, b_guess=b_guess)

        x_t, mu, lam, r = reduced_compiled.construct_gurobi_model(model)
        x = compiled.construct_gurobi_model(model, only_primal=True)

        obj = 0
        for i in range(compiled.P.shape[0]):
            for j in range(compiled.P.shape[1]):
                obj += 0.5 * x[i] * compiled.P[i, j] * x[j]
            obj += compiled.q[i] * x[i]
        for i in range(reduced_compiled.P.shape[0]):
            for j in range(reduced_compiled.P.shape[1]):
                obj -= 0.5 * x_t[i] * reduced_compiled.P[i, j] * x_t[j]
            obj -= reduced_compiled.q[i] * x_t[i]

        model.setObjective(obj, GRB.MINIMIZE)
        model.update()
        model.optimize()

        return model.objBound, [p.x for p in parameters]

    def _constrain_parameters(self, model, parameters):
        if isinstance(self.parameter_set, Polytope):
            A, b = self.parameter_set.A, self.parameter_set.b
            for i in range(A.shape[0]):
                model.addConstr(LinExpr(A[i, :], parameters) <= b[i])
        elif isinstance(self.parameter_set, Box):
            lb, ub = self.parameter_set.lb, self.parameter_set.ub
            for i in range(lb.shape[0]):
                model.addConstr(lb[i] <= parameters[i])
                model.addConstr(parameters[i] <= ub[i])
        model.update()

    def _connect_problem(self, problem, model, parameters, variables, guess=None):
        if isinstance(problem, NNProcessor):
            parameter_size = problem.parameter_size
            variable_size = problem.variable_size

            neurons = {}
            problem.construct_gurobi_model(model=model, neurons=neurons, guess=guess)

            for i in range(parameter_size):
                model.addConstr(parameters[i] == neurons[-1][i])
            for i in range(variable_size):
                model.addConstr(variables[i] == neurons[problem.n_layers - 1][i])
            model.update()

        elif isinstance(problem, QPProblem):
            parameter_size = problem.parameter_size()

            compiled = CompiledQPProblem(problem.problem())

            A_par = np.zeros((parameter_size, compiled.A.shape[1]))
            b_par = np.zeros(parameter_size, dtype=object)
            if guess is None:
                b_guess = None
            else:
                b_guess = np.zeros(parameter_size)
            i = 0
            for j in range(len(problem.parameters())):
                idx = compiled.var_id_to_col[problem.parameters()[j].id]
                size = problem.parameters()[j].size
                for k in range(size):
                    A_par[i, idx + k] = 1
                    b_par[i] = parameters[i]
                    if guess is not None:
                        b_guess[i] = guess[i]
                    i += 1
            compiled.add_eq_constraints(A_par, b_par, b_guess)

            x, mu, lam, r = compiled.construct_gurobi_model(model)

            i = 0
            for j in range(len(problem.variables())):
                idx = compiled.var_id_to_col[problem.variables()[j].id]
                size = problem.variables()[j].size
                for k in range(size):
                    model.addConstr(variables[i] == x[idx + k])
                    i += 1
            model.update()
