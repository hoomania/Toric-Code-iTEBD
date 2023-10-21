from ncon import ncon
from scipy.linalg import expm
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import time


class iTEBD:
    # lambda_b, gamma_a, lambda_a, gamma_b, lambda_b
    MPS_NODE_INDICES = [
        [3, 0, 1, 2, 3],
        [1, 2, 3, 0, 1],
    ]
    MPS_CONTRACT_LEGS_INDICES = [
        [-1, 1],
        [1, 2, 3],
        [3, 4],
        [4, 5, 6],
        [6, -2],
        [2, 5, -3, -4],
    ]
    MPS_CONTRACT_FINAL_ORDER = [-1, -3, -4, -2]

    EXPECTATION_MPS_CONTRACT_LEG_INDICES = [
        [-1, 1],
        [1, -2, 2],
        [2, 3],
        [3, -3, 4],
        [4, -4],
    ]
    EXPECTATION_CONTRACT_LEGS_INDICES = [
        [1, 2, 3, 4],
        [2, 3, 5, 6],
        [1, 5, 6, 4]
    ]

    def __init__(
            self,
            hamiltonian: dict,
            physical_dim: int,
            virtual_dim: int = 2,
            hamiltonian_title: str = 'untitled'
    ):
        self.hamil = hamiltonian
        self.hamil_title = hamiltonian_title
        self.phy_dim = physical_dim
        self.vir_dim = virtual_dim
        self.phy_vir_dim = self.phy_dim * self.vir_dim
        self.delta = 0  # trotter_delta

    def suzuki_trotter(
            self,
            delta: float
    ) -> list:
        output = []
        keys = ['AB', 'BA']
        for key in keys:
            hamil_shape = self.hamil[key].shape
            reshape_dim = hamil_shape[0] * hamil_shape[1]
            power = -delta * np.reshape(self.hamil[key], [reshape_dim, reshape_dim])
            output.append(expm(power).reshape(hamil_shape))

        return output

    def initial_mps_nodes(
            self,
            unit_cells: int = 1
    ) -> list:
        nodes = []
        for i in range(0, unit_cells * 2):
            gamma = np.random.rand(self.vir_dim, self.phy_dim, self.vir_dim)
            nodes.append(gamma / np.max(np.abs(gamma)))
            lambda_ = np.random.rand(self.vir_dim)
            nodes.append(np.diag(lambda_ / sum(lambda_)))

        return nodes

    def cell_update(
            self,
            mps_chain_cell: np.ndarray,
            trotter_tensor: list,
            unit_cells_per_mps: int,
            odd_even_indexes: np.ndarray
    ) -> np.ndarray:
        # odd_even_indexes = self.even_odd_index_generator(unit_cells_per_mps)
        tensor_chain = [0 for _ in range(6)]
        for i in range(2):
            for uc in range(unit_cells_per_mps):
                steps = i % 2

                pointer = odd_even_indexes[steps][uc]
                for j in range(5):
                    tensor_chain[j] = mps_chain_cell[pointer[j]]
                tensor_chain[5] = trotter_tensor[steps]

                tensor_contraction = ncon(tensor_chain, self.MPS_CONTRACT_LEGS_INDICES, None,
                                          self.MPS_CONTRACT_FINAL_ORDER)
                # implode
                implode = np.reshape(tensor_contraction, [self.phy_vir_dim, self.phy_vir_dim])

                # SVD decomposition
                svd_u, svd_sig, svd_v = np.linalg.svd(implode)

                # SVD truncate
                mps_chain_cell[pointer[1]] = np.reshape(
                    svd_u[:, :self.vir_dim],
                    [self.vir_dim, self.phy_dim, self.vir_dim]
                )

                mps_chain_cell[pointer[2]] = np.diag(
                    svd_sig[:self.vir_dim] / sum(svd_sig[:self.vir_dim])
                )

                mps_chain_cell[pointer[3]] = np.reshape(
                    svd_v[:self.vir_dim, :],
                    [self.vir_dim, self.phy_dim, self.vir_dim]
                )

                inverse_l_nodes = 1 / np.diag(mps_chain_cell[pointer[0]])
                inverse_r_nodes = 1 / np.diag(mps_chain_cell[pointer[4]])

                mps_chain_cell[pointer[1]] = ncon(
                    [np.diag(inverse_l_nodes), mps_chain_cell[pointer[1]]],
                    [[-1, 1], [1, -2, -3]])
                mps_chain_cell[pointer[3]] = ncon(
                    [mps_chain_cell[pointer[3]], np.diag(inverse_r_nodes)],
                    [[-1, -2, 1], [1, -3]])

        return mps_chain_cell

    def expectation_energy(
            self,
            mps_nodes: np.ndarray
    ) -> float:
        tensor_chain = [0 for _ in range(5)]
        expectation_value = [0 for _ in range(2)]

        direction = ['AB', 'BA']
        for i in range(2):
            steps = i % 2

            for j in range(5):
                tensor_chain[j] = mps_nodes[self.MPS_NODE_INDICES[steps][j]]

            contraction = ncon(tensor_chain, self.EXPECTATION_MPS_CONTRACT_LEG_INDICES)
            expectation_value[i] = ncon(
                [contraction, self.hamil[direction[steps]], np.conj(contraction)],
                self.EXPECTATION_CONTRACT_LEGS_INDICES
            )
            norm = ncon(
                [contraction, np.conj(contraction)],
                [self.EXPECTATION_CONTRACT_LEGS_INDICES[0], self.EXPECTATION_CONTRACT_LEGS_INDICES[0]]
            )

            expectation_value[i] /= norm

        return sum(expectation_value) / 2

    def expectation_energy_two_cell(
            self,
            mps_nodes: np.ndarray
    ) -> float:
        tensor_chain = [0 for _ in range(9)]
        expectation_value = [0 for _ in range(2)]

        t_MPS_NODE_INDICES = [
            [7, 0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 0, 1],
        ]

        t_EXPECTATION_MPS_CONTRACT_LEG_INDICES = [
            [-1, 1],
            [1, -2, 2],
            [2, 3],
            [3, -3, 4],
            [4, 5],
            [5, -4, 6],
            [6, 7],
            [7, -5, 8],
            [8, -6],
        ]

        t_EXPECTATION_CONTRACT_LEGS_INDICES = [
            [1, 2, 3, 4, 5, 6],
            [2, 3, 7, 8],
            [4, 5, 9, 10],
            [1, 7, 8, 9, 10, 6],
        ]

        direction = ['AB', 'BA']
        for i in range(2):
            steps = i % 2
            # print(len(mps_nodes))
            for j in range(9):
                tensor_chain[j] = mps_nodes[t_MPS_NODE_INDICES[steps][j]]

            contraction = ncon(tensor_chain, t_EXPECTATION_MPS_CONTRACT_LEG_INDICES)
            expectation_value[i] = ncon(
                [contraction, self.hamil[direction[steps]], self.hamil[direction[steps]], np.conj(contraction)],
                t_EXPECTATION_CONTRACT_LEGS_INDICES
            )
            norm = ncon(
                [contraction, np.conj(contraction)],
                [t_EXPECTATION_CONTRACT_LEGS_INDICES[0], t_EXPECTATION_CONTRACT_LEGS_INDICES[0]]
            )

            expectation_value[i] /= norm

        return sum(expectation_value) / 2

    def expectation_arbitrary_operator(
            self,
            mps_nodes: np.ndarray,
            operator: dict
    ) -> float:
        tensor_chain = [0 for _ in range(5)]
        expectation_value = [0 for _ in range(2)]

        direction = ['AB', 'BA']
        for i in range(2):
            steps = i % 2

            for j in range(5):
                tensor_chain[j] = mps_nodes[self.MPS_NODE_INDICES[steps][j]]

            contraction = ncon(tensor_chain, self.EXPECTATION_MPS_CONTRACT_LEG_INDICES)
            expectation_value[i] = ncon(
                [contraction, operator[direction[steps]], np.conj(contraction)],
                self.EXPECTATION_CONTRACT_LEGS_INDICES
            )
            norm = ncon(
                [contraction, np.conj(contraction)],
                [self.EXPECTATION_CONTRACT_LEGS_INDICES[0], self.EXPECTATION_CONTRACT_LEGS_INDICES[0]]
            )

            expectation_value[i] /= norm

        return sum(expectation_value) / 2

    def expectation_arbitrary_operator_two_cell(
            self,
            mps_nodes: np.ndarray,
            operator: dict
    ) -> float:
        tensor_chain = [0 for _ in range(9)]
        expectation_value = [0 for _ in range(2)]

        t_MPS_NODE_INDICES = [
            [7, 0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 0, 1],
        ]

        t_EXPECTATION_MPS_CONTRACT_LEG_INDICES = [
            [-1, 1],
            [1, -2, 2],
            [2, 3],
            [3, -3, 4],
            [4, 5],
            [5, -4, 6],
            [6, 7],
            [7, -5, 8],
            [8, -6],
        ]

        t_EXPECTATION_CONTRACT_LEGS_INDICES = [
            [1, 2, 3, 4, 5, 6],
            [2, 3, 7, 8],
            [4, 5, 9, 10],
            [1, 7, 8, 9, 10, 6],
        ]

        direction = ['AB', 'BA']
        for i in range(2):
            steps = i % 2
            # print(len(mps_nodes))
            for j in range(9):
                tensor_chain[j] = mps_nodes[t_MPS_NODE_INDICES[steps][j]]

            contraction = ncon(tensor_chain, t_EXPECTATION_MPS_CONTRACT_LEG_INDICES)
            expectation_value[i] = ncon(
                [contraction, operator[direction[steps]], operator[direction[steps]], np.conj(contraction)],
                t_EXPECTATION_CONTRACT_LEGS_INDICES
            )
            norm = ncon(
                [contraction, np.conj(contraction)],
                [t_EXPECTATION_CONTRACT_LEGS_INDICES[0], t_EXPECTATION_CONTRACT_LEGS_INDICES[0]]
            )

            expectation_value[i] /= norm

        return sum(expectation_value) / 2

    def expectation_single_site(
            self,
            mps_nodes: np.ndarray,
            operator: dict
    ) -> float:
        tensor_chain = [0 for _ in range(5)]

        for j in range(5):
            tensor_chain[j] = mps_nodes[self.MPS_NODE_INDICES[0][j]]

        contraction = ncon(tensor_chain, self.EXPECTATION_MPS_CONTRACT_LEG_INDICES)
        expectation_value = ncon(
            [contraction, operator['AB'], np.conj(contraction)],
            self.EXPECTATION_CONTRACT_LEGS_INDICES
        )
        norm = ncon(
            [contraction, np.conj(contraction)],
            [self.EXPECTATION_CONTRACT_LEGS_INDICES[0], self.EXPECTATION_CONTRACT_LEGS_INDICES[0]]
        )

        expectation_value /= norm

        return sum(expectation_value)

    def iTEBD_trotter_manager(
            self,
            iteration: int,
            domains: int,
            delta_start: float = 0.01,
            delta_end: float = 0.0001,
            accuracy: float = 1e-16,
            unit_cells: int = 1
    ) -> np.ndarray:

        result = {
            'dist': np.inf,
            'energy': np.inf,
            'mps': self.initial_mps_nodes(unit_cells)
        }

        if iteration % domains != 0:
            iteration -= iteration % domains

        logfile_prefix = int(time.time())
        iter_value = int(iteration / domains)
        for delta in np.linspace(delta_start, delta_end, domains):
            self.delta = delta
            itebd_result = self.iTEBD_check_(
                result['mps'],
                self.suzuki_trotter(delta),
                iter_value,
                logfile_prefix,
                accuracy,
                unit_cells
            )

            if itebd_result['dist'] < result['dist']:
                result = {
                    'dist': itebd_result['dist'],
                    'energy': itebd_result['energy'],
                    'mps': itebd_result['mps']
                }

        # print(f'Best Energy: {result["energy"]}')
        return result['mps']

    def iTEBD(
            self,
            mps_chain_cell: np.ndarray,
            trotter_tensor: list,
            iteration: int,
            logfile_prefix: int,
            accuracy: float = 1e-8
    ) -> dict:

        # >>>> initial parameters
        tensor_chain = [0 for _ in range(6)]
        expectation_diff = [0, 0]
        sampling = int(iteration * 0.1)
        expectation_energy_history = []
        best_result = {
            'dist': np.inf,
            'energy': 0,
            'mps': []
        }
        # <<<< initial parameters

        prg = tqdm(range(sampling, iteration + sampling + 2), desc=f'iTEBD is running, delta = {self.delta:.5f}',
                   leave=True)
        for i in prg:
            steps = i % 2

            for j in range(5):
                tensor_chain[j] = mps_chain_cell[self.MPS_NODE_INDICES[steps][j]]
            tensor_chain[5] = trotter_tensor[steps]

            tensor_contraction = ncon(tensor_chain, self.MPS_CONTRACT_LEGS_INDICES, None, self.MPS_CONTRACT_FINAL_ORDER)
            # implode
            implode = np.reshape(tensor_contraction, [self.phy_vir_dim, self.phy_vir_dim])

            # SVD decomposition
            svd_u, svd_sig, svd_v = np.linalg.svd(implode)

            # SVD truncate
            mps_chain_cell[self.MPS_NODE_INDICES[steps][1]] = np.reshape(
                svd_u[:, :self.vir_dim],
                [self.vir_dim, self.phy_dim, self.vir_dim]
            )

            mps_chain_cell[self.MPS_NODE_INDICES[steps][2]] = np.diag(
                svd_sig[:self.vir_dim] / sum(svd_sig[:self.vir_dim])
            )

            mps_chain_cell[self.MPS_NODE_INDICES[steps][3]] = np.reshape(
                svd_v[:self.vir_dim, :],
                [self.vir_dim, self.phy_dim, self.vir_dim]
            )

            inverse_lr_nodes = 1 / np.diag(mps_chain_cell[self.MPS_NODE_INDICES[steps][4]])

            mps_chain_cell[self.MPS_NODE_INDICES[steps][1]] = ncon(
                [np.diag(inverse_lr_nodes), mps_chain_cell[self.MPS_NODE_INDICES[steps][1]]],
                [[-1, 1], [1, -2, -3]])
            mps_chain_cell[self.MPS_NODE_INDICES[steps][3]] = ncon(
                [mps_chain_cell[self.MPS_NODE_INDICES[steps][3]], np.diag(inverse_lr_nodes)],
                [[-1, -2, 1], [1, -3]])

            if i % sampling == 0:
                xpc_energy = self.expectation_energy(mps_chain_cell)
                expectation_energy_history.append(xpc_energy)
                expectation_diff[0] = xpc_energy
                if len(expectation_energy_history) != 1:

                    mean_ = np.mean(expectation_energy_history)
                    if np.abs(xpc_energy - mean_) < best_result['dist']:
                        prg.set_postfix_str(f'Best Energy: {xpc_energy:.16f}')
                        prg.refresh()  # to show immediately the update
                        best_result = {
                            'dist': np.abs(xpc_energy - mean_),
                            'energy': xpc_energy,
                            'mps': mps_chain_cell
                        }

            if (i + 1) % sampling == 2:
                expectation_diff[1] = self.expectation_energy(mps_chain_cell)
                if np.abs(expectation_diff[0] - expectation_diff[1]) < accuracy:
                    break

        # logfile = f'mps_data/{logfile_prefix}_history_energy.csv'
        # df = pd.DataFrame(expectation_energy_history)
        # df.to_csv(logfile, mode='a', header=False, index=False)

        return best_result  # mps_chain_cell

    def iTEBD_check_(
            self,
            mps_chain_cell: np.ndarray,
            trotter_tensor: list,
            iteration: int,
            logfile_prefix: int,
            accuracy: float = 1e-8,
            unit_cells: int = 1
    ) -> dict:

        # >>>> initial parameters
        # tensor_chain = [0 for _ in range(6)]
        expectation_diff = [0, 0]
        sampling = int(iteration * 0.1)
        expectation_energy_history = []
        best_result = {
            'dist': np.inf,
            'energy': 0,
            'mps': []
        }
        # <<<< initial parameters

        prg = tqdm(range(sampling, iteration + sampling + 2), desc=f'iTEBD is running, delta = {self.delta:.5f}',
                   leave=True)
        odd_even_indexes = self.even_odd_index_generator(unit_cells)
        for i in prg:

            mps_chain_cell = self.cell_update(
                mps_chain_cell,
                trotter_tensor,
                unit_cells,
                odd_even_indexes
            )

            if i % sampling == 0:
                if unit_cells == 2:
                    xpc_energy = self.expectation_energy_two_cell(mps_chain_cell)
                else:
                    xpc_energy = self.expectation_energy(mps_chain_cell)

                expectation_energy_history.append(xpc_energy)
                expectation_diff[0] = xpc_energy
                if len(expectation_energy_history) != 1:

                    mean_ = np.mean(expectation_energy_history)
                    if np.abs(xpc_energy - mean_) < best_result['dist']:
                        prg.set_postfix_str(f'Best Energy: {xpc_energy:.16f}')
                        prg.refresh()  # to show immediately the update
                        best_result = {
                            'dist': np.abs(xpc_energy - mean_),
                            'energy': xpc_energy,
                            'mps': mps_chain_cell
                        }

            if (i + 1) % sampling == 2:
                if unit_cells == 2:
                    expectation_diff[1] = self.expectation_energy_two_cell(mps_chain_cell)
                else:
                    expectation_diff[1] = self.expectation_energy(mps_chain_cell)

                if np.abs(expectation_diff[0] - expectation_diff[1]) < accuracy:
                    break

        # logfile = f'mps_data/{logfile_prefix}_history_energy.csv'
        # df = pd.DataFrame(expectation_energy_history)
        # df.to_csv(logfile, mode='a', header=False, index=False)

        return best_result  # mps_chain_cell

    def save_data(
            self,
            data_list: list,
            final_iteration: int,
            accuracy: float
    ) -> None:
        file_name_capsule = f'{int(time.time())}_history_capsule.csv'
        file_name_element = f'{int(time.time())}_history_element.csv'
        file_name_energy = f'{int(time.time())}_history_energy.csv'

        data_profile = {
            'hamiltonian_title': [self.hamil_title],
            'phy_dim': [self.phy_dim],
            'vir_dim': [self.vir_dim],
            'trotter_delta': [self.delta],
            'final_iteration': [final_iteration],
            'accuracy': [accuracy],
            'file_name_capsule': [file_name_capsule],
            'file_name_element': [file_name_element],
            'file_name_energy': [file_name_energy],
        }

        df = pd.DataFrame(data_profile)
        output_path = 'mps_data/data_profile.csv'
        df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

        df = pd.DataFrame(data_list[0])
        df.to_csv(f'mps_data/{file_name_capsule}', header=False, index=False)

        df = pd.DataFrame(data_list[1])
        df.to_csv(f'mps_data/{file_name_element}', header=False, index=False)

        df = pd.DataFrame(data_list[2])
        df.to_csv(f'mps_data/{file_name_energy}', header=False, index=False)

    def iTEBD_diagram(
            self,
            iteration: int,
            accuracy: float = 1e-8,
            delta: float = 0.01,
    ) -> dict:
        # >>>> initial parameters
        mps_chain_cell = self.initial_mps_nodes()
        exp_tensor = self.suzuki_trotter(delta)

        # lambda_b, gamma_a, lambda_a, gamma_b, lambda_b
        chain_indices = [
            [3, 0, 1, 2, 3],
            [1, 2, 3, 0, 1],
        ]

        tensor_chain = [0 for _ in range(6)]
        links = [
            [-1, 1],
            [1, 2, 3],
            [3, 4],
            [4, 5, 6],
            [6, -2],
            [2, -3, -4, 5],
        ]
        final_order = [-1, -3, -4, -2]

        expectation_diff = [0, 0]
        expectation_value = []
        # <<<< initial parameters

        final_iteration = 0
        ph_vi_dimension = self.phy_dim * self.vir_dim
        chain_capsule_history = []
        chain_element_history = []
        # for i in range(iteration):
        for i in tqdm(range(iteration), ncols=100, desc="iTEBD is running: "):
            steps = i % 2

            for j in range(5):
                tensor_chain[j] = mps_chain_cell[chain_indices[steps][j]]
            tensor_chain[5] = exp_tensor[steps]

            chain_element_row = np.reshape(tensor_chain[0], (-1,))
            for w in range(1, 5):
                chain_element_row = np.hstack([chain_element_row, np.reshape(tensor_chain[w], (-1,))])
            chain_element_history.append(chain_element_row)

            tensor_contraction = ncon(tensor_chain, links, None, final_order)
            chain_capsule_history.append(tensor_contraction.reshape(-1, ))
            # implode
            implode = np.reshape(tensor_contraction, [ph_vi_dimension, ph_vi_dimension])

            # SVD decomposition
            svd_u, svd_sig, svd_v = np.linalg.svd(implode)

            # SVD truncate
            mps_chain_cell[chain_indices[steps][1]] = np.reshape(
                svd_u[:, :self.vir_dim],
                [self.vir_dim, self.phy_dim, self.vir_dim]
            )

            mps_chain_cell[chain_indices[steps][2]] = np.diag(
                svd_sig[:self.vir_dim] / sum(svd_sig[:self.vir_dim])
            )

            mps_chain_cell[chain_indices[steps][3]] = np.reshape(
                svd_v[:self.vir_dim, :],
                [self.vir_dim, self.phy_dim, self.vir_dim]
            )

            inverse_lr_nodes = 1 / np.diag(mps_chain_cell[chain_indices[steps][4]])

            mps_chain_cell[chain_indices[steps][1]] = ncon(
                [np.diag(inverse_lr_nodes), mps_chain_cell[chain_indices[steps][1]]],
                [[-1, 1], [1, -2, -3]])
            mps_chain_cell[chain_indices[steps][3]] = ncon(
                [mps_chain_cell[chain_indices[steps][3]], np.diag(inverse_lr_nodes)],
                [[-1, -2, 1], [1, -3]])

            exp_value = self.expectation_energy(mps_chain_cell)
            expectation_value.append(exp_value)
            expectation_diff[i % 2] = exp_value
            if i % 10000 == 0:
                # print(f'Iteration: {i}')
                if np.abs(expectation_diff[0] - expectation_diff[1]) < accuracy:
                    final_iteration = i
                    break

        self.save_data(
            [chain_capsule_history, chain_element_history, expectation_value],
            final_iteration,
            accuracy
        )

        return {
            'iteration': final_iteration,
            # 'energy_expectation_value': expectation_value,
            'mps_chain': mps_chain_cell,
        }

    def even_odd_index_generator(
            self,
            unit_cells_per_mps: int
    ) -> np.ndarray:
        len_mps = unit_cells_per_mps * 4
        len_itebd_cell = int(len_mps / unit_cells_per_mps)

        indexes = [i % len_mps for i in range(len_mps - 1, len_mps * 2)]
        start_index = [0, 2]  # even, odd
        start_index_itebd = [len_itebd_cell * i for i in range(unit_cells_per_mps)]
        index_divider = []
        for p in range(2):
            for q in range(unit_cells_per_mps):
                index = []
                for j in range(len_itebd_cell + 1):
                    index.append(indexes[(start_index_itebd[q] + start_index[p] + j) % len_mps])
                index_divider.append(index)
        return np.reshape(index_divider, (2, unit_cells_per_mps, len_itebd_cell + 1))
