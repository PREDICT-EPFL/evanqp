import os.path
import argparse
import pickle
import numpy as np
import cvxpy as cp
import polytope as pc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from evanqp import MPCProblem, Polytope, RandomSampler, Verifier
from evanqp.layers import BoundArithmetic
from utils import dlqr


torch.manual_seed(0)
np.random.seed(0)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class DoubleIntegrator(MPCProblem):
    def __init__(self, N=10):
        self.N = N

        n = 2
        m = 1

        # Dynamics
        self.A = np.array([[1, 1], [0, 1]])
        self.B = np.array([[0], [1]])
        self.C = np.eye(n)

        # Weights
        self.Q = np.diag([1.0, 1.0])
        self.R = np.diag([0.1])
        self.K, self.P, _ = dlqr(self.A, self.B, self.Q, self.R)

        # Constraints
        self.x_max = np.array([10.0, 10.0])
        self.x_min = np.array([-10.0, -10.0])
        self.u_max = 1.0
        self.u_min = -1.0

        # Terminal Set computation
        # state constraints
        Hx = np.vstack((np.eye(n), -np.eye(n)))
        hx = np.concatenate((self.x_max, -self.x_min))
        # input constraints
        Hu = np.vstack((np.eye(m), -np.eye(m)))
        hu = np.array([self.u_max, -self.u_min])
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

        objective = cp.quad_form(self.x[N, :], self.P)
        constraints = [self.x0 == self.x[0, :], self.u0 == self.u[0, :]]

        for i in range(N):
            objective += cp.quad_form(self.x[i, :], self.Q) + cp.quad_form(self.u[i, :], self.R)
            constraints += [self.x[i + 1, :] == self.A @ self.x[i, :] + self.B @ self.u[i, :]]
            constraints += [self.x_min <= self.x[i, :], self.x[i, :] <= self.x_max]
            constraints += [self.u_min <= self.u[i, :], self.u[i, :] <= self.u_max]
        constraints += [self.F @ self.x[N, :] <= self.f]

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

        solution = {self.u0: None,
                    self.u: None,
                    self.x: None,
                    self.objective: None}

        try:
            self.prob.solve(solver=cp.GUROBI, warm_start=True)
            solution = {self.u0: self.u0.value,
                        self.u: self.u.value,
                        self.x: self.x.value,
                        self.objective: self.objective.value}
        except:
            pass

        return solution

    def dynamics(self):
        return self.A, self.B

    def reduced_objective(self):
        objective = cp.quad_form(self.x[self.N, :], self.P)
        for i in range(1, self.N):
            objective += cp.quad_form(self.x[i, :], self.Q) + cp.quad_form(self.u[i, :], self.R)
        return cp.Minimize(objective)


class DoubleIntegratorDataset(Dataset):
    def __init__(self, parameter_samples, variable_samples):
        self.parameter_samples = parameter_samples
        self.variable_samples = variable_samples

    def __len__(self):
        return self.parameter_samples.shape[0]

    def __getitem__(self, idx):
        return self.parameter_samples[idx, :], self.variable_samples[idx, :]


def loss_function(model, data, target):
    output = model(data)
    loss = F.mse_loss(output, target)
    loss += 1e2 / len(data) * F.mse_loss(model(torch.zeros(2).float().to(data.device)), torch.zeros(1).float().to(data.device))
    return loss


def train(model, device, train_loader, optimizer):
    model.train()
    train_loss = 0
    norm_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        loss = loss_function(model, data, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        norm_loss = max(torch.linalg.norm(model(data) - target, ord=np.inf), norm_loss)

    train_loss /= len(train_loader)
    print(f'Train Loss: {train_loss:.6f}, Norm Loss: {norm_loss:.6f}')


def train_bfgs(model, device, train_loader, optimizer):
    model.train()
    train_loss = 0
    norm_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        def closure():
            optimizer.zero_grad()
            loss = loss_function(model, data, target)
            loss.backward()
            return loss

        optimizer.step(closure)

        train_loss += closure()
        norm_loss = max(torch.linalg.norm(model(data) - target, ord=np.inf), norm_loss)

    train_loss /= len(train_loader)
    print(f'Train Loss: {train_loss:.6f}, Norm Loss: {norm_loss:.6f}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            test_loss += loss_function(model, data, target).item()

    test_loss /= len(test_loader)
    print(f'Test set: Average loss: {test_loss:.6f}')


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='DoubleIntegrator Lipschitz Benchmark')
    parser.add_argument('--samples', type=int, default=int(1e4))
    parser.add_argument('--hidden-layers', type=int, default=2)
    parser.add_argument('--hidden-dimensions', type=int, default=18)
    parser.add_argument('--threads', type=int, default=0)
    args = parser.parse_args()

    mpc_controller = DoubleIntegrator()

    parameter_set = Polytope(np.vstack((np.eye(2), -np.eye(2))), np.concatenate((mpc_controller.x_max, -mpc_controller.x_min)))

    parameter_samples = None
    variable_samples = None
    if os.path.isfile('parameter_samples.pt'):
        parameter_samples = torch.load('parameter_samples.pt')
        variable_samples = torch.load('variable_samples.pt')

    if parameter_samples is None or parameter_samples.shape[0] < args.samples:
        sampler = RandomSampler(mpc_controller, parameter_set)
        parameter_samples, variable_samples = sampler.sample(args.samples, seed=1)

        parameter_samples = torch.from_numpy(parameter_samples).float()
        variable_samples = torch.from_numpy(variable_samples).float()

        torch.save(parameter_samples, 'parameter_samples.pt')
        torch.save(variable_samples, 'variable_samples.pt')

    parameter_samples = parameter_samples[0:args.samples, :]
    variable_samples = variable_samples[0:args.samples, :]

    # fig, ax = plt.subplots()
    # cs = ax.tricontourf(parameter_samples[:, 0].numpy(), parameter_samples[:, 1].numpy(), variable_samples[:, 0].numpy(), levels=100)
    # ax.scatter(parameter_samples[:, 0].numpy(), parameter_samples[:, 1].numpy(), c='black', s=1)
    # fig.colorbar(cs)
    # ax.set_xlabel(r'$x_1$')
    # ax.set_ylabel(r'$x_2$')
    # plt.show(block=False)

    dataset = DoubleIntegratorDataset(parameter_samples, variable_samples)

    train_set_ratio = 0.8
    train_set_size = int(len(dataset) * train_set_ratio)
    test_set_size = len(dataset) - train_set_size
    train_set, test_set = random_split(dataset, [train_set_size, test_set_size], generator=torch.Generator().manual_seed(1))

    hidden_dim = args.hidden_dimensions
    net = nn.Sequential(
        nn.Linear(mpc_controller.parameter_size(), hidden_dim),
        nn.ReLU(),
        *[layer for layer_pairs in [[nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] for _ in range(args.hidden_layers - 1)] for layer in layer_pairs],
        nn.Linear(hidden_dim, mpc_controller.variable_size()),
    )

    use_cuda = torch.cuda.is_available()
    use_cuda = False
    device = torch.device('cuda' if use_cuda else 'cpu')

    if args.threads != 0:
        torch.set_num_threads(args.threads)

    train_kwargs = {'batch_size': 1024}
    test_kwargs = {'batch_size': 1000}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(dataset, **train_kwargs)
    # train_loader = DataLoader(train_set, **train_kwargs)
    # test_loader = DataLoader(test_set, **test_kwargs)

    net.to(device)

    optimizer = optim.LBFGS(net.parameters(), max_iter=25, tolerance_grad=1e-7, lr=0.1, line_search_fn='strong_wolfe')
    for epoch in range(1, 10 + 1):
        train_bfgs(net, device, train_loader, optimizer)

    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(1, 500 + 1):
        train(net, device, train_loader, optimizer)

    # x1 = np.linspace(-10, 10, 100)
    # x2 = np.linspace(-6, 6, 100)
    # x, y = np.meshgrid(x1, x2)
    # u1_opt = np.zeros(x.shape)
    # for i in tqdm(range(x.shape[0])):
    #     for j in range(x.shape[1]):
    #         sol = mpc_controller.solve(np.array([x[i, j], y[i, j]]))
    #         u0 = sol[mpc_controller.variables()[0]]
    #         u1_opt[i, j] = u0[0] if u0 is not None else None
    #
    # u_net = np.vstack((x.flatten(), y.flatten())).T
    # u_net = net(torch.from_numpy(u_net).float())
    # u_net = u_net.detach().cpu().numpy()
    # u1_net = u_net[:, 0].reshape(x.shape)
    #
    # fig, ax = plt.subplots()
    # cs = ax.contourf(x, y, u1_opt, levels=100)
    # fig.colorbar(cs)
    # ax.set_xlabel(r'$x_1$')
    # ax.set_ylabel(r'$x_2$')
    # plt.show(block=False)
    #
    # fig, ax = plt.subplots()
    # cs = ax.contourf(x, y, u1_net, levels=100)
    # fig.colorbar(cs)
    # ax.set_xlabel(r'$x_1$')
    # ax.set_ylabel(r'$x_2$')
    # plt.show(block=False)
    #
    # fig, ax = plt.subplots()
    # cs = ax.contourf(x, y, np.abs(u1_opt - u1_net), levels=100)
    # fig.colorbar(cs)
    # ax.set_xlabel(r'$x_1$')
    # ax.set_ylabel(r'$x_2$')
    # plt.show()

    data = {
        'samples': args.samples,
        'hidden_layers': args.hidden_layers,
        'hidden-dimensions': args.hidden_dimensions,
        'threads': args.threads,
    }

    verifier = Verifier(parameter_set, mpc_controller, net)
    verifier.compute_bounds(method=BoundArithmetic.ZONO_ARITHMETIC)

    bound, parameters = verifier.approximation_error(threads=args.threads)
    data['approximation_error'] = {
        'Runtime': verifier.model.Runtime,
        'MIPGap': verifier.model.MIPGap,
        'Bound': bound,
        'Parameters': parameters,
    }

    bound, parameters = verifier.verify_stability_sufficient(threads=args.threads)
    data['verify_stability_sufficient'] = {
        'Runtime': verifier.model.Runtime,
        'MIPGap': verifier.model.MIPGap,
        'Bound': bound,
        'Parameters': parameters,
    }

    bound, parameters = verifier.verify_stability_direct(threads=args.threads)
    data['verify_stability_direct'] = {
        'Runtime': verifier.model.Runtime,
        'MIPGap': verifier.model.MIPGap,
        'Bound': bound,
        'Parameters': parameters,
    }

    terminal_set = Polytope(mpc_controller.F, mpc_controller.f)
    verifier = Verifier(terminal_set, mpc_controller, net)
    verifier.compute_bounds(method=BoundArithmetic.ZONO_ARITHMETIC)
    verifier.compute_bounds_lipschitz()
    bound, gain_mpc, gain_nn, parameters = verifier.approximation_error_lipschitz_constant()
    data['approximation_error_lipschitz_constant'] = {
        'Runtime': verifier.model.Runtime,
        'MIPGap': verifier.model.MIPGap,
        'Bound': bound,
        'Parameters': parameters,
    }

    print(data)
    pickle.dump(data, open(f'benchmark_data/{args.samples}_{args.hidden_layers}_{args.hidden_dimensions}.p', 'wb'))


if __name__ == '__main__':
    main()
