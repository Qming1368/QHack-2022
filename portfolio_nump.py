
from qiskit import Aer
from qiskit.algorithms import VQE, QAOA, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA,ADAM
from qiskit.circuit.library import TwoLocal
from qiskit.utils import QuantumInstance
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_finance.data_providers import RandomDataProvider
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications import OptimizationApplication
from qiskit_optimization.converters import QuadraticProgramToQubo
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time


def portfolio_optimization(num_assets):

    seed = 1
    stocks = [("TICKER%s" % i) for i in range(num_assets)]
    data = RandomDataProvider(tickers=stocks,
                     start=datetime.datetime(2016,1,1),
                     end=datetime.datetime(2016,1,30),seed=seed)


    data.run()
    mu = data.get_period_return_mean_vector()
    sigma = data.get_period_return_covariance_matrix()
    q = 0.5                   # set risk factor
    budget = num_assets // 2  # set budget
    penalty = num_assets      # set parameter to scale the budget penalty term


    portfolio = PortfolioOptimization(expected_returns=mu, covariances=sigma, risk_factor=q, budget=budget)
    qp = portfolio.to_quadratic_program()


    from qiskit.utils import algorithm_globals

    algorithm_globals.massive=True
    algorithm_globals.random_seed = 1234

    exact_mes = NumPyMinimumEigensolver()
    exact_eigensolver = MinimumEigenOptimizer(exact_mes)
    start = time.time()
    result = exact_eigensolver.solve(qp)

    end = time.time()

    selection = result.x


    return end-start, selection

if __name__ == "__main__":
    runtime_list = []
    selection_list = []
    for num_assets in range(2,29):
        runtime, selection = portfolio_optimization(num_assets)
        print(
            "num_assets: {} | Runtime: {:0.2f} | Selection: {} ".format(
                num_assets, runtime,[int(i) for i in selection]
            )
        )

        runtime_list.append(runtime)
        selection_list.append(selection)
    np.save('numpy_runtime.npy',np.array(runtime_list))
    np.save('numpy_solution.npy',np.array(selection_list,dtype=object))
