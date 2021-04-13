import numpy as np
from tqdm import tqdm

from evanqp.sets import Polytope, Box
from evanqp.problems import QPProblem
from evanqp.utils import cprnd


class RandomSampler:

    def __init__(self, problem, parameter_set):
        if not isinstance(problem, QPProblem):
            raise Exception('problem must be of type QPProblem')

        self.problem = problem
        self.parameter_set = parameter_set

    def sample(self, samples, seed=None):
        if seed is not None:
            np.random.seed(seed)

        if isinstance(self.parameter_set, Polytope):
            n = self.parameter_set.A.shape[1]
        elif isinstance(self.parameter_set, Box):
            n = self.parameter_set.lb.shape[0]
        else:
            raise Exception('parameter_set type not supported.')
        m = sum([v.size for v in self.problem.variables()])

        sample_buffer_size = min(samples, 1000)

        def sample_input_batch():
            if isinstance(self.parameter_set, Polytope):
                return cprnd(sample_buffer_size, self.parameter_set.A, self.parameter_set.b)
            if isinstance(self.parameter_set, Box):
                lb = self.parameter_set.lb
                ub = self.parameter_set.ub
                s = np.random.rand(sample_buffer_size, lb.shape[0])
                lb_rep = np.tile(lb, (sample_buffer_size, 1))
                ub_rep = np.tile(ub, (sample_buffer_size, 1))
                s = lb_rep + (ub_rep - lb_rep) * s
                return s

        parameter_samples = np.empty((samples, n))
        variable_samples = np.empty((samples, m))
        with tqdm(total=samples) as t:
            i = 0
            j = 0
            inp_samples = sample_input_batch()
            while i < samples:
                sol = self.problem.solve(inp_samples[j, :])
                if sol[self.problem.variables()[0]] is not None:
                    parameter_samples[i, :] = inp_samples[j, :]
                    variable_samples[i, :] = np.concatenate([sol[v] for v in self.problem.variables()])
                    i += 1
                    t.update()
                j += 1
                if j >= sample_buffer_size:
                    inp_samples = sample_input_batch()
                    j = 0

        return parameter_samples, variable_samples
