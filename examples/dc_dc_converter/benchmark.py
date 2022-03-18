import argparse
import pickle
import numpy as np
import cvxpy as cp
import polytope as pc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from evanqp import MPCProblem, Polytope, RandomSampler, Verifier
from evanqp.layers import BoundArithmetic, SeqLayer, LinearLayer, ReluLayer
from utils import dlqr


torch.manual_seed(0)
np.random.seed(0)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class DCDCConverter(MPCProblem):
    def __init__(self, N=10):
        self.N = N

        n = 2
        m = 1

        # Linearized dynamics
        self.A = np.array([[0.971356387900839, -0.009766890567613], [1.731870774203751, 0.970462385352837]])
        self.B = np.array([[0.148778899882612], [0.180827260808426]])
        self.C = np.array([[0, 1]])

        # Steady state
        ref = 5.0
        ss = np.linalg.solve(np.block([[self.A - np.eye(n), self.B], [self.C, 0]]), np.array([0, 0, ref]))
        self.xs = ss[0:2]
        self.us = ss[2:3]

        # Weights
        self.Q = np.diag([90, 1])
        self.R = np.array([[1]])
        self.K, self.P, _ = dlqr(self.A, self.B, self.Q, self.R)

        # Constraints
        self.x_max = np.array([0.2, 7.0])
        self.x_min = np.array([0.0, 0.0])
        self.u_max = np.array([1.0])
        self.u_min = np.array([0.0])

        # Terminal Set computation (shifted)
        # state constraints
        Hx = np.vstack((np.eye(n), -np.eye(n)))
        hx = np.concatenate((self.x_max - self.xs, -(self.x_min - self.xs)))
        # input constraints
        Hu = np.vstack((np.eye(m), -np.eye(m)))
        hu = np.concatenate((self.u_max - self.us, -(self.u_min - self.us)))
        # closed loop dynamics
        Ak = self.A - self.B @ self.K
        # state & input constraints
        HH = np.vstack((Hx, -Hu @ self.K))
        hh = np.concatenate((hx, hu))
        # compute maximal invariant set
        O = pc.Polytope(HH, hh)
        while True:
            O_prev = O
            # pre-set
            O = O.intersect(pc.Polytope(O.A @ Ak, O.b))
            if O == O_prev:
                break
        self.F, self.f = O.A, O.b

        self.x0 = cp.Parameter(n, name='x0')
        self.x = cp.Variable((N + 1, n), name='x')
        self.u0 = cp.Variable(m, name='u0')
        self.u = cp.Variable((N, m), name='u')

        objective = cp.quad_form(self.x[N, :] - self.xs, self.P)
        constraints = [self.x0 == self.x[0, :], self.u0 == self.u[0, :]]

        for i in range(N):
            objective += cp.quad_form(self.x[i, :] - self.xs, self.Q) + cp.quad_form(self.u[i, :] - self.us, self.R)
            constraints += [self.x[i + 1, :] == self.A @ self.x[i, :] + self.B @ self.u[i, :]]
            constraints += [self.x_min <= self.x[i, :], self.x[i, :] <= self.x_max]
            constraints += [self.u_min <= self.u[i, :], self.u[i, :] <= self.u_max]
        constraints += [self.F @ (self.x[N, :] - self.xs) <= self.f]

        self.objective = cp.Minimize(objective)
        self.prob = cp.Problem(self.objective, constraints)

    def problem(self):
        return self.prob

    def parameters(self):
        return [self.x0]

    def variables(self):
        return [self.u0]

    def solve(self, x0):
        self.x0.value = x0
        try:
            self.prob.solve(solver=cp.MOSEK)
        except:
            pass

        solution = {self.u0: self.u0.value,
                    self.u: self.u.value,
                    self.x: self.x.value,
                    self.objective: self.objective.value}
        return solution

    def dynamics(self):
        return self.A, self.B

    def reduced_objective(self):
        objective = cp.quad_form(self.x[self.N, :] - self.xs, self.P)
        for i in range(1, self.N):
            objective += cp.quad_form(self.x[i, :] - self.xs, self.Q) + cp.quad_form(self.u[i, :] - self.us, self.R)
        return cp.Minimize(objective)


class SatModule(nn.Module):
    def __init__(self, num_features, x_min, x_max):
        super(SatModule, self).__init__()
        self.num_features = num_features
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        return torch.max(torch.min(x, torch.from_numpy(self.x_max).float()), torch.from_numpy(self.x_min).float())

    def milp_layer(self, depth):
        return SeqLayer([
            LinearLayer(np.eye(self.num_features), -self.x_min, depth),
            ReluLayer(self.num_features, depth),
            LinearLayer(np.eye(self.num_features), self.x_min, depth),
            LinearLayer(-np.eye(self.num_features), self.x_max, depth),
            ReluLayer(self.num_features, depth),
            LinearLayer(-np.eye(self.num_features), self.x_max, depth)
        ], depth)


class DCDCConverterDataset(Dataset):
    def __init__(self, parameter_samples, variable_samples):
        self.parameter_samples = parameter_samples
        self.variable_samples = variable_samples

    def __len__(self):
        return self.parameter_samples.shape[0]

    def __getitem__(self, idx):
        return self.parameter_samples[idx, :], self.variable_samples[idx, :]


def loss_function(model, data, target, mpc_controller):
    output = model(data)
    loss = F.mse_loss(output, target)
    loss += 1e2 / len(data) * F.mse_loss(model(torch.from_numpy(mpc_controller.xs).float().to(data.device)), torch.from_numpy(mpc_controller.us).float().to(data.device))
    return loss


def train(model, device, train_loader, optimizer, mpc_controller):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        loss = loss_function(model, data, target, mpc_controller)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='DC-DC Converter Benchmark')
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--hidden-layers', type=int, default=1)
    parser.add_argument('--hidden-dimensions', type=int, default=50)
    parser.add_argument('--threads', type=int, default=0)
    args = parser.parse_args()

    mpc_controller = DCDCConverter(N=args.horizon)

    parameter_set = Polytope(np.vstack((np.eye(2), -np.eye(2))), np.concatenate((mpc_controller.x_max, -mpc_controller.x_min)))

    sampler = RandomSampler(mpc_controller, parameter_set)
    parameter_samples, variable_samples = sampler.sample(args.samples, seed=1)

    parameter_samples = torch.from_numpy(parameter_samples).float()
    variable_samples = torch.from_numpy(variable_samples).float()

    dataset = DCDCConverterDataset(parameter_samples, variable_samples)

    hidden_dim = args.hidden_dimensions
    net = nn.Sequential(
        nn.Linear(mpc_controller.parameter_size(), hidden_dim),
        nn.ReLU(),
        *[layer for layer_pairs in [[nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] for _ in range(args.hidden_layers)] for layer in layer_pairs],
        nn.Linear(hidden_dim, mpc_controller.variable_size()),
    )

    use_cuda = torch.cuda.is_available()

    device = torch.device('cuda' if use_cuda else 'cpu')

    if args.threads != 0:
        torch.set_num_threads(args.threads)

    train_kwargs = {'batch_size': 50}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(dataset, **train_kwargs)

    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(1, 250 + 1):
        train(net, device, train_loader, optimizer, mpc_controller)

    net_sat = nn.Sequential(
        *net,
        SatModule(mpc_controller.variable_size(), mpc_controller.u_min, mpc_controller.u_max)
    )

    data = {
        'horizon': args.horizon,
        'samples': args.samples,
        'hidden_layers': args.hidden_layers,
        'hidden-dimensions': args.hidden_dimensions,
        'threads': args.threads,
    }

    verifier = Verifier(parameter_set, mpc_controller, net_sat)
    verifier.compute_bounds(method=BoundArithmetic.ZONO_ARITHMETIC)

    print(verifier.approximation_error(guess=mpc_controller.xs, threads=args.threads))
    data['find_max_abs_diff'] = {
        'Runtime': verifier.model.Runtime,
        'MIPGap': verifier.model.MIPGap,
    }

    print(verifier.verify_stability_sufficient(guess=mpc_controller.xs, threads=args.threads))
    data['verify_stability_sufficient'] = {
        'Runtime': verifier.model.Runtime,
        'MIPGap': verifier.model.MIPGap,
    }

    print(verifier.verify_stability_direct(guess=mpc_controller.xs, threads=args.threads))
    data['verify_stability_direct'] = {
        'Runtime': verifier.model.Runtime,
        'MIPGap': verifier.model.MIPGap,
    }

    print(data)
    pickle.dump(data, open(f'benchmark_data/{args.horizon}_{args.hidden_layers}_{args.hidden_dimensions}.p', 'wb'))


if __name__ == '__main__':
    main()
