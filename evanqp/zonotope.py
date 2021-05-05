import numpy as np


class Zonotope:

    def __init__(self, head, errors):
        self.head = head
        self.errors = errors

    @staticmethod
    def zonotope_from_box(box):
        lb, ub = box.lb, box.ub
        center = 0.5 * (ub + lb)
        beta = 0.5 * (ub - lb)
        errors = np.diag(beta)
        return Zonotope(center, errors)

    @property
    def shape(self):
        return self.head.shape

    def __sub__(self, other):
        return Zonotope(self.head - other, self.errors)

    def __add__(self, other):
        return Zonotope(self.head + other, self.errors)

    def __truediv__(self, other):
        return Zonotope(self.head / other, self.errors / other)

    def __matmul__(self, other):
        return Zonotope(self.head @ other, self.errors @ other)

    def copy(self):
        return Zonotope(np.copy(self.head), np.copy(self.errors))

    def concretize(self):
        delta = np.sum(np.abs(self.errors), axis=0)
        return self.head - delta, self.head + delta

    def linear(self, weight, bias):
        return self @ weight.T + bias

    def relu(self, bounds=None):
        lb, ub = self.concretize()
        if bounds is not None:
            lb_refined, ub_refined = bounds
            lb = np.max(lb, lb_refined)
            ub = np.min(ub, ub_refined)

        D = 1e-6
        is_cross = (lb < 0) & (ub > 0)
        relu_lambda = np.where(is_cross, ub/(ub-lb+D), (lb >= 0).astype(float))
        relu_mu = np.where(is_cross, -0.5 * ub * lb / (ub - lb + D), np.zeros(lb.shape))

        assert (not np.any(np.isnan(relu_mu))) and (not np.any(np.isnan(relu_lambda)))

        new_head = self.head * relu_lambda + relu_mu
        old_errs = self.errors * relu_lambda
        new_errs = Zonotope._get_new_errs(is_cross, new_head, relu_mu)
        new_errors = np.concatenate((old_errs, new_errs), axis=0)

        assert (not np.any(np.isnan(new_head))) and (not np.any(np.isnan(new_errors)))

        return Zonotope(new_head, new_errors)

    @staticmethod
    def _get_new_errs(should_box, new_head, new_beta):
        new_err_pos = should_box > 0
        num_new_errs = int(np.sum(new_err_pos))
        n = new_head.shape[0]
        beta_values = new_beta[new_err_pos]
        new_errs = np.zeros((num_new_errs, n))
        new_errs[np.arange(num_new_errs), new_err_pos] = beta_values
        return new_errs
