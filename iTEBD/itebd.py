from iTEBD import hamiltonian as hamil
from ncon import ncon
from scipy.linalg import expm
from tqdm import tqdm
import numpy as np
import re


class iTEBD:

    def __init__(
            self,
            hamiltonian: dict,
            physical_dim: int,
            virtual_dim: int = 2,
            unit_cells: int = 1
    ):
        self.hamil = hamiltonian
        self.phy_dim = physical_dim
        self.vir_dim = virtual_dim
        self.unit_cells = unit_cells
        self.matrix_type = hamiltonian['matrix_type']

        self.mps = self.initial_mps()
        self.phy_vir_dim = self.phy_dim * self.vir_dim
        self.delta = 0
        self.accuracy = 1e-16

        self.MPS_NODE_INDICES = self.indices_mps_node()
        self.EXPECTATION_MPS_CONTRACT_LEG_INDICES = self.expectation_mps_contract_leg_indices()
        self.EXPECTATION_CONTRACT_LEGS_INDICES = self.expectation_contract_legs_indices()
        self.EXPECTATION_NORM_LEGS_INDICES = self.expectation_norm_legs_indices()
        self.EXPECTATION_CONTRACT_LEGS_ONE_UNIT_CELL_INDICES = self.expectation_contract_legs_one_unit_cell_indices()
        self.MPS_CONTRACT_LEGS_INDICES = [
            [-1, 1],
            [1, 2, 3],
            [3, 4],
            [4, 5, 6],
            [6, -4],
            [2, 5, -2, -3],
        ]
        self.INDICES_EVEN_ODD_BOND = self.indices_even_odd_bond()

    def indices_mps_node(self):
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

    def indices_even_odd_bond(self) -> np.ndarray:
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

    def initial_mps(self) -> list:
        nodes = []
        for i in range(0, self.unit_cells * 2):
            gamma = np.random.rand(self.vir_dim, self.phy_dim, self.vir_dim)
            nodes.append(gamma / np.max(np.abs(gamma)))
            lambda_ = np.random.rand(self.vir_dim)
            nodes.append(np.diag(lambda_ / sum(lambda_)))

        return nodes

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

    def delta_manager(
            self,
            iteration: int,
            delta_steps: int,
            delta_start: float = 0.01,
            delta_end: float = 0.0001,
            accuracy: float = 1e-16,
    ) -> list:

        if iteration % delta_steps != 0:
            iteration -= iteration % delta_steps

        iter_value = int(iteration / delta_steps)
        self.accuracy = accuracy

        print(f'iTEBD is running... \nPhysical Dim: {self.phy_dim} \nBond Dim: {self.vir_dim}')
        for self.delta in np.linspace(delta_start, delta_end, delta_steps):
            self.evolution(
                self.suzuki_trotter(self.delta),
                iter_value
            )

        return self.mps

    def evolution(
            self,
            trotter_tensor: list,
            iteration: int
    ) -> None:

        # >>>> initial parameters
        sampling = int(iteration * 0.1)
        expectation_diff = [0, 0]
        expectation_energy_history = []
        best_distance = np.inf
        # <<<< initial parameters

        prg = tqdm(range(sampling, iteration + sampling + 2), desc=f'delta= {self.delta:.5f}',
                   leave=True)

        for i in prg:

            self.mps = self.cell_update(trotter_tensor)

            if i % sampling == 0:
                xpc_energy = sum(self.mps_bonds_energy(self.mps, self.hamil))

                expectation_energy_history.append(xpc_energy)
                expectation_diff[0] = xpc_energy
                if len(expectation_energy_history) != 1:

                    mean_energy = np.mean(expectation_energy_history)
                    if np.abs(xpc_energy - mean_energy) < best_distance:
                        prg.set_postfix_str(f'Best Energy: {xpc_energy:.16f}')
                        prg.refresh()  # to show immediately the update
                        best_distance = np.abs(xpc_energy - mean_energy)

            if (i + 1) % sampling == 2:
                expectation_diff[1] = sum(self.mps_bonds_energy(self.mps, self.hamil))

                if np.abs(expectation_diff[0] - expectation_diff[1]) < self.accuracy:
                    break

    def cell_update(
            self,
            trotter_tensor: list,
    ) -> list:
        tensor_chain = [0 for _ in range(6)]
        for i in range(2):
            for uc in range(self.unit_cells):

                pointer = self.INDICES_EVEN_ODD_BOND[i][uc]

                for j in range(5):
                    tensor_chain[j] = self.mps[pointer[j]]
                tensor_chain[5] = trotter_tensor[i]

                tensor_contraction = ncon(tensor_chain, self.MPS_CONTRACT_LEGS_INDICES)
                # implode
                implode = np.reshape(tensor_contraction, [self.phy_vir_dim, self.phy_vir_dim])

                # SVD decomposition
                svd_u, svd_sig, svd_v = np.linalg.svd(implode)

                # SVD truncate
                self.mps[pointer[1]] = np.reshape(
                    svd_u[:, :self.vir_dim],
                    [self.vir_dim, self.phy_dim, self.vir_dim]
                )

                self.mps[pointer[2]] = np.diag(
                    svd_sig[:self.vir_dim] / sum(svd_sig[:self.vir_dim])
                )

                self.mps[pointer[3]] = np.reshape(
                    svd_v[:self.vir_dim, :],
                    [self.vir_dim, self.phy_dim, self.vir_dim]
                )

                inverse_l_nodes = 1 / np.diag(self.mps[pointer[0]])
                inverse_r_nodes = 1 / np.diag(self.mps[pointer[4]])

                self.mps[pointer[1]] = ncon(
                    [np.diag(inverse_l_nodes), self.mps[pointer[1]]],
                    [[-1, 1], [1, -2, -3]])
                self.mps[pointer[3]] = ncon(
                    [self.mps[pointer[3]], np.diag(inverse_r_nodes)],
                    [[-1, -2, 1], [1, -3]])

        return self.mps

    def mps_bonds_energy(
            self,
            mps: list,
            operator: dict
    ) -> list:
        expectation_value = []

        direction = ['AB', 'BA']
        for i in range(2):
            for j in range(self.unit_cells):
                expectation_value.append(
                    self.expectation_bond(
                        mps,
                        operator[direction[i]],
                        (2 * j) + (i + 1)
                    )
                )

        # return sum(expectation_value)
        return expectation_value

    def expectation_bond(
            self,
            mps: list,
            operator: dict,
            bond_index: int,
    ) -> float:
        index = bond_index - 1
        mps = [mps[i % len(mps)] for i in range(4 * index, len(mps) + (4 * index))]

        tensor_chain = []
        for j in range((self.unit_cells * 4) + 1):
            tensor_chain.append(mps[self.MPS_NODE_INDICES[0][j]])

        for j in range((self.unit_cells * 4) + 1):
            tensor_chain.append(np.conj(mps[self.MPS_NODE_INDICES[0][j]]))

        norm = ncon(
            tensor_chain,
            self.expectation_norm_legs_indices()
        )

        tensor_chain.append(operator)

        return ncon(
            tensor_chain,
            self.EXPECTATION_CONTRACT_LEGS_ONE_UNIT_CELL_INDICES
        ) / norm

    def expectation_single_site_mag(
            self,
            mps: list,
            unit_cell_index: int,
            site_index: int,
            magnetization: str
    ) -> float:
        if unit_cell_index != 1:
            index = unit_cell_index - 1
            mps = [mps[i % len(mps)] for i in range(4 * index, len(mps) + (4 * index))]

        tensor_chain = []
        for j in range((self.unit_cells * 4) + 1):
            tensor_chain.append(mps[self.MPS_NODE_INDICES[0][j]])

        for j in range((self.unit_cells * 4) + 1):
            tensor_chain.append(np.conj(mps[self.MPS_NODE_INDICES[0][j]]))

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
        operator = hamil.Hamiltonian(matrix_type=self.matrix_type).encode_hamil([hamil_str])
        tensor_chain.append(operator['AB'])

        return ncon(
            tensor_chain,
            self.EXPECTATION_CONTRACT_LEGS_ONE_UNIT_CELL_INDICES
        ) / norm

    def expectation_all_sites_mag(
            self,
            mps: list,
            mag_direction: str = 'xyz'
    ) -> dict:
        mags = np.unique(re.findall('[xyz]', mag_direction))
        sites = int(2 * np.log2(self.phy_dim))

        mag_dict = {
            'x': [],
            'y': [],
            'z': [],
            'mean_x': 0,
            'mean_y': 0,
            'mean_z': 0,
            'mag_value': 0
        }

        for mag in mags:
            for unit_cell in range(self.unit_cells):
                for site in range(1, sites + 1):
                    mag_dict[mag].append(self.expectation_single_site_mag(
                        mps,
                        unit_cell,
                        site,
                        mag
                    ))

            mag_dict[f'mean_{mag}'] = np.mean(mag_dict[mag])

        mag_dict['mag_value'] = np.sqrt(mag_dict['mean_x']**2 + mag_dict['mean_y']**2 + mag_dict['mean_z']**2)

        return mag_dict
