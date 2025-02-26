from qiskit import Aer, QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_machine_learning.algorithms import QSVM

class QuantumML:
    def __init__(self):
        self.backend = Aer.get_backend('qasm_simulator')
        self.qc = QuantumCircuit(2)
        self.theta = Parameter('θ')
        self.qc.h(0)
        self.qc.cx(0, 1)
        self.qc.ry(self.theta, 1)

    def train_qsvm(self, X_train, y_train):
        qsvm = QSVM(self.qc, X_train, y_train)
        qsvm.fit()
        return qsvm