

from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_finance.data_providers import RandomDataProvider
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications import OptimizationApplication
from qiskit_optimization.converters import QuadraticProgramToQubo
import datetime
import time
import pennylane as qml
from pennylane import qaoa
from pennylane import numpy as np


def portfolio_optimization(num_assets):

    stocks = [("TICKER%s" % i) for i in range(num_assets)]
    data = RandomDataProvider(tickers=stocks,
                     start=datetime.datetime(2016,1,1),
                     end=datetime.datetime(2016,1,30),seed=1)


    data.run()
    mu = data.get_period_return_mean_vector()
    sigma = data.get_period_return_covariance_matrix()

    q = 0.5                   # set risk factor
    budget = num_assets // 2  # set budget
    penalty = num_assets      # set parameter to scale the budget penalty term

    portfolio = PortfolioOptimization(expected_returns=mu, covariances=sigma, risk_factor=q, budget=budget)
    qp = portfolio.to_quadratic_program()
    qubo = QuadraticProgramToQubo().convert(qp)
    qubitOp, offset = qubo.to_ising()
    coeffs = []
    obs_list = []
    for i in qubitOp.to_pauli_op():
        po_list = str(i).split(' * ')
        coeffs.append(float(po_list[0]))
        Z_indices = [j for j, c in enumerate(po_list[1]) if c == 'Z']
        if len(Z_indices) == 1:
            obs_list.append(qml.PauliZ(num_assets-1-Z_indices[0]))
        if len(Z_indices) == 2:
            obs_list.append(qml.PauliZ(num_assets-1-Z_indices[0])@ qml.PauliZ(num_assets-1-Z_indices[1]))
    cost_h = qml.Hamiltonian(coeffs, obs_list)

    coeffs_mixed = [1]*num_assets
    obs_mixed_list = []
    for i in range(num_assets):
        obs_mixed_list.append(qml.PauliX(i))
    mixer_h = qml.Hamiltonian(coeffs_mixed, obs_mixed_list)


    dev = qml.device('default.qubit', wires=num_assets)
    wires = range(num_assets)
    depth = 3
    steps = 1000
    optimizer = qml.AdamOptimizer()
    def qaoa_layer(gamma, alpha):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(alpha, mixer_h)

    def circuit(params, **kwargs):
        for w in wires:
            qml.Hadamard(wires=w)
        qml.layer(qaoa_layer, depth, params[0], params[1])
    @qml.qnode(dev)
    def result_circuit(params, **kwargs):
        for w in wires:
            qml.Hadamard(wires=w)
        qml.layer(qaoa_layer, depth, params[0], params[1])
        return qml.probs(wires=wires)

    @qml.qnode(dev,diff_method="best")
    def cost_function(params):
        circuit(params)
        return qml.expval(cost_h)

    params = 0.01 * np.random.randn(2,depth,requires_grad=True)

    start = time.time()
    cost_pre = 0
    for i in range(steps):
        params, cost = optimizer.step_and_cost(cost_function, params)
        if np.abs(cost-cost_pre) < 1e-06:
            break
        cost_pre = np.copy(cost)
        #print("Cost function: ",cost+offset)
    end = time.time()


    probs = result_circuit(params)

    selection = np.binary_repr(np.argmax(probs),width=num_assets)


    return end-start, selection

if __name__ == "__main__":
    runtime_list = []
    selection_list = []
    for num_assets in range(2,18):
        runtime, selection = portfolio_optimization(num_assets)
        print(
            "num_assets: {} | Runtime: {:0.2f} | Selection: {} ".format(
                num_assets, runtime,[int(i) for i in selection]
            )
        )

        runtime_list.append(runtime)
        selection_list.append(selection)
    np.save('pennylane_runtime.npy',np.array(runtime_list))
    np.save('pennylane_solution.npy',np.array(selection_list,dtype=object))
