from abc import ABC, abstractmethod
from typing import List
import cvxpy as cp
from cvxpy.problems.objective import Objective


class QPProblem(ABC):

    @abstractmethod
    def problem(self) -> cp.Problem:
        pass

    @abstractmethod
    def parameters(self) -> List[cp.Parameter]:
        pass

    @abstractmethod
    def variables(self) -> List[cp.Variable]:
        pass

    @abstractmethod
    def solve(self, *args) -> dict:
        pass

    def parameter_size(self):
        return sum([v.size for v in self.parameters()])

    def variable_size(self):
        return sum([v.size for v in self.variables()])


class MPCProblem(QPProblem):

    @abstractmethod
    def reduced_objective(self) -> Objective:
        pass
