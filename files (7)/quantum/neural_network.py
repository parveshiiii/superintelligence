import pennylane as qml
from typing import List, Optional
import numpy as np

class QuantumNeuralNetwork:
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.quantum_circuit = qml.QNode(self._circuit, self.dev)
        self.params = self._initialize_parameters()

    def _circuit(self, inputs: np.ndarray, params: np.ndarray) -> np.ndarray:
        # Encode inputs
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)

        # Apply variational layers
        param_idx = 0
        for _ in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.Rot(*params[param_idx:param_idx + 3], wires=i)
                param_idx += 3
            
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.quantum_circuit(inputs, self.params)