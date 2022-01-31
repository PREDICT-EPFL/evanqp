from abc import ABC, abstractmethod
from typing import List, Union, Tuple
import numpy as np
import cvxpy as cp


class CvxpyProblem(ABC):

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

    def parameter_size(self) -> int:
        return sum([v.size for v in self.parameters()])

    def variable_size(self) -> int:
        return sum([v.size for v in self.variables()])


class MPCProblem(CvxpyProblem, ABC):

    def dynamics(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError('dynamics() not implemented on MPCProblem, should return (A,B).')

    def reduced_objective(self) -> Union[cp.Minimize, cp.Maximize]:
        raise NotImplementedError('reduced_objective() not implemented on MPCProblem, should return objective without first stage cost.')
