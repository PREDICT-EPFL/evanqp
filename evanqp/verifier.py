import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from gurobipy import GRB, Model, LinExpr, abs_, max_
from tqdm import tqdm

from evanqp import SeqNet, QPProblem
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

    def construct_gurobi_model(self, model=None, neurons=None, milp_tightening=False, lp_relaxation=False):
        if isinstance(self.parameter_set, Box):
            lb = torch.from_numpy(self.parameter_set.lb).float()
            ub = torch.from_numpy(self.parameter_set.ub).float()
        elif isinstance(self.parameter_set, Polytope):
            lb = torch.from_numpy(self.parameter_set.largest_interior_box().lb).float()
            ub = torch.from_numpy(self.parameter_set.largest_interior_box().ub).float()
        else:
            raise NotImplementedError()

        if model is not None:
            neurons[-1] = [model.addVar(vtype=GRB.CONTINUOUS, lb=lb.numpy()[i], ub=ub.numpy()[i], name=f'in_{i}') for i in range(self.parameter_size)]
            if isinstance(self.parameter_set, Polytope):
                A, b = self.parameter_set.A, self.parameter_set.b
                for i in range(A.shape[0]):
                    model.addConstr(LinExpr(A[i, :], neurons[-1]) <= b[i])
            model.update()

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

    P = data['P'].toarray()
    q = data['q']
    A = data['A'].toarray()
    b = data['b']
    F = data['F'].toarray()
    g = data['G']

    restore_params(objective, param_dic)
    for con in constraints:
        restore_params(con, param_dic)

    return P, q, A, b, F, g, compiler.var_id_to_col


class CompiledQPProblem:

    def __init__(self, problem):
        self.problem = problem

        P, q, A, b, F, g, var_id_to_col = compile_qp_problem_with_params_as_variables(problem.problem())
        self.P = P
        self.q = q
        self.A = A
        self.b = b
        self.F = F
        self.g = g
        self.var_id_to_col = var_id_to_col

    def add_eq_constraints(self, A, b):
        self.A = np.vstack((self.A, A))
        self.b = np.concatenate((self.b, b))

    def construct_gurobi_model(self, model):
        x = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'x_{i}') for i in range(self.P.shape[1])]
        mu = [model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'mu_{i}') for i in range(self.A.shape[0])]
        lam = [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name=f'lam_{i}') for i in range(self.F.shape[0])]
        model.update()

        for i in range(self.P.shape[0]):
            model.addConstr(LinExpr(self.P[i, :], x) + self.q[i] + LinExpr(self.A.T[i, :], mu) + LinExpr(self.F.T[i, :], lam) == 0)
        for i in range(self.A.shape[0]):
            model.addConstr(LinExpr(self.A[i, :], x) - self.b[i] == 0)
        for i in range(self.F.shape[0]):
            model.addConstr(LinExpr(self.F[i, :], x) - self.g[i] <= 0)
        model.update()

        r = [model.addVar(vtype=GRB.BINARY, name=f'r_{i}') for i in range(self.F.shape[0])]
        model.update()
        for i in range(self.F.shape[0]):
            # M = 1e5
            # model.addConstr(F[i, :] @ x - g[i] >= -r[i] * M)
            # model.addConstr(lam[i] <= (1 - r[i]) * M)
            model.addConstr((r[i] == 0) >> (LinExpr(self.F[i, :], x) - self.g[i] == 0))
            model.addConstr((r[i] == 1) >> (lam[i] == 0))
        model.update()

        return x, mu, lam, r


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

    def find_max_abs_diff(self, threads=0):
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

        self._connect_problems(self.ref_problem, model, parameters, ref_variables)
        self._connect_problems(self.approx_problem, model, parameters, approx_variables)

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

    def verify_stability(self, threads=0):
        pass

    def _connect_problems(self, problem, model, parameters, variables):
        if isinstance(problem, NNProcessor):
            parameter_size = problem.parameter_size
            variable_size = problem.variable_size

            neurons = {}
            problem.construct_gurobi_model(model=model, neurons=neurons)

            for i in range(parameter_size):
                model.addConstr(parameters[i] == neurons[-1][i])
            for i in range(variable_size):
                model.addConstr(variables[i] == neurons[problem.n_layers - 1][i])
            model.update()

        elif isinstance(problem, QPProblem):
            parameter_size = problem.parameter_size()

            compiled = CompiledQPProblem(problem)

            A_par = np.zeros((parameter_size, compiled.A.shape[1]))
            b_par = np.zeros(parameter_size, dtype=object)
            i = 0
            for j in range(len(problem.parameters())):
                idx = compiled.var_id_to_col[problem.parameters()[j].id]
                size = problem.parameters()[j].size
                for k in range(size):
                    A_par[i, idx + k] = 1
                    b_par[i] = parameters[i]
                    i += 1
            compiled.add_eq_constraints(A_par, b_par)

            x, mu, lam, r = compiled.construct_gurobi_model(model)

            i = 0
            for j in range(len(problem.variables())):
                idx = compiled.var_id_to_col[problem.variables()[j].id]
                size = problem.variables()[j].size
                for k in range(size):
                    model.addConstr(variables[i] == x[idx + k])
                    i += 1
            model.update()
