import numpy as np


class Operator:
    def __init__(self):
        self.pauli_dict = {
            'x': np.array([[0, 1], [1, 0]]),
            'y': np.array([[0, -1j], [1j, 0]]),
            'z': np.array([[1, 0], [0, -1]]),
            'i': np.array([[1, 0], [0, 1]]),
        }

        self.spin_dict = {
            'x': 0.5 * self.pauli_dict['x'],
            'y': 0.5 * self.pauli_dict['y'],
            'z': 0.5 * self.pauli_dict['z'],
            'i': np.array([[1, 0], [0, 1]]),
        }

    def pauli_dictionary(self) -> dict:
        return self.pauli_dict

    def spin_dictionary(self) -> dict:
        return self.spin_dict

    def pauli_matrix(self, key: str) -> np.ndarray:
        return self.pauli_dict[key]

    def identity(self) -> np.ndarray:
        return self.pauli_dict['i']

    def pauli_x(self) -> np.ndarray:
        return self.pauli_dict['x']

    def pauli_y(self) -> np.ndarray:
        return self.pauli_dict['y']

    def pauli_z(self) -> np.ndarray:
        return self.pauli_dict['z']

    def spin_matrix(self, key: str) -> np.ndarray:
        return self.pauli_dict[key]

    def spin_x(self) -> np.ndarray:
        return self.spin_dict['x']

    def spin_y(self) -> np.ndarray:
        return self.spin_dict['y']

    def spin_z(self) -> np.ndarray:
        return self.spin_dict['z']
