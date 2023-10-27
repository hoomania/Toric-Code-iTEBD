from ncon import ncon
from scipy.linalg import expm
from tqdm import tqdm
import numpy as np
import hamiltonian as hamil


class iTEBD:
    # lambda_b, gamma_a, lambda_a, gamma_b, lambda_b
    # MPS_NODE_INDICES = [
    #     [3, 0, 1, 2, 3],
    #     [1, 2, 3, 0, 1],
    # ]
    # # MPS_CONTRACT_LEGS_INDICES = [
    # #     [-1, 1],
    # #     [1, 2, 3],
    # #     [3, 4],
    # #     [4, 5, 6],
    # #     [6, -2],
    # #     [2, 5, -3, -4],
    # # ]
    #
    # MPS_CONTRACT_LEGS_INDICES = [
    #     [-1, 1],
    #     [1, 2, 3],
    #     [3, 4],
    #     [4, 5, 6],
    #     [6, -4],
    #     [2, 5, -2, -3],
    # ]
    #
    # # MPS_CONTRACT_FINAL_ORDER = [-1, -3, -4, -2]
    #
    # EXPECTATION_MPS_CONTRACT_LEG_INDICES = [
    #     [-1, 1],
    #     [1, -2, 2],
    #     [2, 3],
    #     [3, -3, 4],
    #     [4, -4],
    # ]
    # EXPECTATION_CONTRACT_LEGS_INDICES = [
    #     [1, 2, 3, 4],
    #     [2, 3, 5, 6],
    #     [1, 5, 6, 4]
    # ]

    def __init__(
            self,
            hamiltonian: dict,
            physical_dim: int,
            virtual_dim: int = 2,
            unit_cells: int = 1,
    ):
        self.hamil = hamiltonian
        self.phy_dim = physical_dim
        self.vir_dim = virtual_dim
        self.phy_vir_dim = self.phy_dim * self.vir_dim
        self.unit_cells = unit_cells

        self.delta = 0
        self.accuracy = 1e-16

        self.mps_nodes = self.initial_mps_nodes()

        self.MPS_NODE_INDICES = self.mps_node_indices()
        self.EXPECTATION_MPS_CONTRACT_LEG_INDICES = self.expectation_mps_contract_leg_indices()
        self.EXPECTATION_CONTRACT_LEGS_INDICES = self.expectation_contract_legs_indices()
        self.MPS_CONTRACT_LEGS_INDICES = [
            [-1, 1],
            [1, 2, 3],
            [3, 4],
            [4, 5, 6],
            [6, -4],
            [2, 5, -2, -3],
        ]
        self.EXPECTATION_NORM_LEGS_INDICES = self.expectation_norm_legs_indices()
        self.EXPECTATION_CONTRACT_LEGS_ONE_UNIT_CELL_INDICES = self.expectation_contract_legs_one_unit_cell_indices()

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

    def initial_mps_nodes(self) -> list:
        nodes = []
        for i in range(0, self.unit_cells * 2):
            gamma = np.random.rand(self.vir_dim, self.phy_dim, self.vir_dim)
            nodes.append(gamma / np.max(np.abs(gamma)))
            lambda_ = np.random.rand(self.vir_dim)
            nodes.append(np.diag(lambda_ / sum(lambda_)))

        return nodes

    def cell_update(
            self,
            mps_chain_cell: np.ndarray,
            trotter_tensor: list,
            odd_even_indexes: np.ndarray
    ) -> np.ndarray:
        tensor_chain = [0 for _ in range(6)]
        for i in range(2):
            for uc in range(self.unit_cells):
                steps = i % 2

                pointer = odd_even_indexes[steps][uc]
                for j in range(5):
                    tensor_chain[j] = mps_chain_cell[pointer[j]]
                tensor_chain[5] = trotter_tensor[steps]

                tensor_contraction = ncon(tensor_chain, self.MPS_CONTRACT_LEGS_INDICES)
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

    def delta_manager(
            self,
            iteration: int,
            domains: int,
            delta_start: float = 0.01,
            delta_end: float = 0.0001,
            accuracy: float = 1e-16,
    ) -> np.ndarray:

        result = {
            'mps': self.initial_mps_nodes(),
            # 'mps': self.mps_nodes,
            'dist': np.inf,
            'energy': np.inf,
        }

        if iteration % domains != 0:
            iteration -= iteration % domains

        iter_value = int(iteration / domains)

        print(f'iTEBD is running on physical dim={self.phy_dim} and virtual dim={self.vir_dim}')
        for delta in np.linspace(delta_start, delta_end, domains):
            self.delta = delta
            self.accuracy = accuracy
            evo_result = self.evolution(
                result['mps'],
                self.suzuki_trotter(delta),
                iter_value
            )

            if evo_result['dist'] < result['dist']:
                result = {
                    'mps': evo_result['mps'],
                    'dist': evo_result['dist'],
                    'energy': evo_result['energy'],
                    'energy_history': evo_result['energy_history']
                }

        return result['mps']

    def mps_node_indices(self):
        len_mps = self.unit_cells * 4
        indices = [
            [i % len_mps for i in range(len_mps - 1, len_mps * 2)],
            [i % len_mps for i in range(1, len_mps + 2)],
        ]
        return indices

    def expectation_mps_contract_leg_indices(self):
        indices_list = [[-1, 1]]
        positive = 1
        ppo = 0
        negative = -2
        for i in range(1, self.unit_cells * 4):
            if i % 2 == 1:
                ppo = positive + 1
                indices_list.append([positive, negative, ppo])
                positive += 2
                negative -= 1
            else:
                indices_list.append([ppo, positive])

        indices_list.append([ppo, negative])
        return indices_list

    def expectation_contract_legs_indices(self) -> list:
        indices_list = []
        positive = 1
        for w in range(2):
            for i in range(0, (self.unit_cells * 4) + 1):
                if i % 2 == 0:
                    indices_list.append([positive, positive + 1])
                    positive += 1
                else:
                    indices_list.append([positive, positive + 1, positive + 2])
                    positive += 2

        indices_list[(self.unit_cells * 4) + 1][0] = indices_list[0][0]
        indices_list[-1][-1] = indices_list[self.unit_cells * 4][-1]

        j = (self.unit_cells * 4) + 1
        for i in range(self.unit_cells):
            t = 1 + (4 * i)
            indices_list.append(
                [
                    indices_list[t][1],
                    indices_list[t + 2][1],
                    indices_list[t + j][1],
                    indices_list[t + 2 + j][1],
                ]
            )

        return indices_list

    def expectation_contract_legs_one_unit_cell_indices(self) -> list:
        indices_list = []
        positive = 1
        for w in range(2):
            for i in range(0, (self.unit_cells * 4) + 1):
                if i % 2 == 0:
                    indices_list.append([positive, positive + 1])
                    positive += 1
                else:
                    indices_list.append([positive, positive + 1, positive + 2])
                    positive += 2

        indices_list[(self.unit_cells * 4) + 1][0] = indices_list[0][0]
        indices_list[-1][-1] = indices_list[self.unit_cells * 4][-1]

        j = (self.unit_cells * 4) + 1
        indices_list.append(
            [
                indices_list[1][1],
                indices_list[3][1],
                indices_list[1 + j][1],
                indices_list[3 + j][1],
            ]
        )
        for i in range(1, self.unit_cells):
            t = 1 + (4 * i)
            indices_list[t + j][1] = indices_list[t][1]
            indices_list[t + j + 2][1] = indices_list[t + 2][1]

        return indices_list

    def expectation_norm_legs_indices(self) -> list:
        indices_list = []
        positive = 1
        for w in range(2):
            for i in range(0, (self.unit_cells * 4) + 1):
                if i % 2 == 0:
                    indices_list.append([positive, positive + 1])
                    positive += 1
                else:
                    indices_list.append([positive, positive + 1, positive + 2])
                    positive += 2

        indices_list[(self.unit_cells * 4) + 1][0] = indices_list[0][0]
        indices_list[-1][-1] = indices_list[self.unit_cells * 4][-1]

        j = (self.unit_cells * 4) + 1
        for i in range(self.unit_cells):
            t = 1 + (4 * i)
            indices_list[t + j][1] = indices_list[t][1]
            indices_list[t + j + 2][1] = indices_list[t + 2][1]

        return indices_list

    def even_odd_index_generator(self) -> np.ndarray:
        len_mps = self.unit_cells * 4
        indexes = [i % len_mps for i in range(len_mps - 1, len_mps * 2)]
        start_index = [0, 2]  # even, odd
        start_index_mps = [4 * i for i in range(self.unit_cells)]

        index_divider = []
        for p in range(2):
            for q in range(self.unit_cells):
                index = []
                for j in range(5):
                    index.append(indexes[(start_index_mps[q] + start_index[p] + j) % len_mps])
                index_divider.append(index)

        return np.reshape(index_divider, (2, self.unit_cells, 5))

    def evolution(
            self,
            mps_chain_cell: np.ndarray,
            trotter_tensor: list,
            iteration: int
    ) -> dict:

        # >>>> initial parameters
        expectation_diff = [0, 0]
        sampling = int(iteration * 0.1)
        expectation_energy_history = []
        best_result = {
            'dist': np.inf,
            'energy': 0,
            'mps': []
        }
        odd_even_indexes = self.even_odd_index_generator()
        # <<<< initial parameters

        prg = tqdm(range(sampling, iteration + sampling + 2), desc=f'delta= {self.delta:.5f}',
                   leave=True)

        for i in prg:

            mps_chain_cell = self.cell_update(
                mps_chain_cell,
                trotter_tensor,
                odd_even_indexes
            )

            if i % sampling == 0:
                xpc_energy = self.expectation_value(mps_chain_cell, self.hamil)

                expectation_energy_history.append(xpc_energy)
                expectation_diff[0] = xpc_energy
                if len(expectation_energy_history) != 1:

                    mean_ = np.mean(expectation_energy_history)
                    if np.abs(xpc_energy - mean_) < best_result['dist']:
                        prg.set_postfix_str(f'Best Energy: {xpc_energy:.16f}')
                        prg.refresh()  # to show immediately the update
                        best_result = {
                            'mps': mps_chain_cell,
                            'dist': np.abs(xpc_energy - mean_),
                            'energy': xpc_energy,
                            'energy_history': expectation_energy_history,
                        }

            if (i + 1) % sampling == 2:
                expectation_diff[1] = self.expectation_value(mps_chain_cell, self.hamil)

                if np.abs(expectation_diff[0] - expectation_diff[1]) < self.accuracy:
                    break

        return best_result

    def expectation_value(
            self,
            mps_nodes: np.ndarray,
            operator: dict
    ) -> float:
        expectation_value = [0, 0]

        direction = ['AB', 'BA']
        for i in range(2):
            steps = i % 2
            tensor_chain = []

            for j in range((self.unit_cells * 4) + 1):
                tensor_chain.append(mps_nodes[self.MPS_NODE_INDICES[steps][j]])

            for j in range((self.unit_cells * 4) + 1):
                tensor_chain.append(np.conj(mps_nodes[self.MPS_NODE_INDICES[steps][j]]))

            norm = ncon(
                tensor_chain,
                self.EXPECTATION_NORM_LEGS_INDICES
            )

            for w in range(self.unit_cells):
                tensor_chain.append(operator[direction[steps]])

            expectation_value[i] = ncon(
                tensor_chain,
                self.EXPECTATION_CONTRACT_LEGS_INDICES
            )

            expectation_value[i] /= norm

        return sum(expectation_value)

    def expectation_single_site_mag(
            self,
            mps_nodes: np.ndarray,
            unit_cell_index: int,
            site_index: int,
            magnetization: str
    ) -> float:
        if unit_cell_index != 1:
            index = unit_cell_index - 1
            mps_nodes = [mps_nodes[i % len(mps_nodes)] for i in range(4 * index, len(mps_nodes) + (4 * index))]

        tensor_chain = []
        for j in range((self.unit_cells * 4) + 1):
            tensor_chain.append(mps_nodes[self.MPS_NODE_INDICES[0][j]])

        for j in range((self.unit_cells * 4) + 1):
            tensor_chain.append(np.conj(mps_nodes[self.MPS_NODE_INDICES[0][j]]))

        norm = ncon(
            tensor_chain,
            self.expectation_norm_legs_indices()
        )

        str_len = int(2 * np.log2(self.phy_dim))
        hamil_str = ''
        for i in range(str_len):
            if site_index - 1 == i:
                hamil_str += magnetization
            else:
                hamil_str += 'i'
        operator = hamil.Hamiltonian().encode_hamil([hamil_str])
        tensor_chain.append(operator['AB'])

        return ncon(
            tensor_chain,
            self.EXPECTATION_CONTRACT_LEGS_ONE_UNIT_CELL_INDICES
        ) / norm

