from iTEBD import hamiltonian as hml, itebd as tebd
import numpy as np
import pandas as pd
import time


class DataSample:
    def __init__(self, dir_path: str):
        time_prefix = time.strftime("%Y_%m_%d_%H_%M_%S_UTC", time.gmtime())
        self.physical_data = f'{dir_path}/{time_prefix}_phy_data_example.csv'
        self.mps_data = f'{dir_path}/{time_prefix}_mps_data_example.csv'

    def save_mps(self, mps: list) -> None:
        mps_row = []
        for item in mps:
            if len(item.shape) == 3:
                mps_row = np.hstack([mps_row, np.reshape(item, (-1,))])
            else:
                mps_row = np.hstack([mps_row, np.diag(item)])

        df = pd.DataFrame(mps_row.reshape(1, mps_row.shape[0]))
        df.to_csv(self.mps_data, mode='a', header=False, index=False)

    def data_sampling_1uc(
            self,
            matrix_type: str = 'pauli',
            virtual_dim: int = 2,
            unit_cells: int = 1,
            hx_min: float = 0.0,
            hx_max: float = 1.4,
            hx_steps: float = 0.05,
            iteration: int = 3000,
            delta_start: int = 0.1,
            delta_end: int = 0.01,
            delta_steps: int = 3,
    ):
        phy_dim = 8
        hamil = hml.Hamiltonian(matrix_type=matrix_type)
        hx_values = np.linspace(hx_min, hx_max, num=int((hx_max - hx_min) / hx_steps))

        for h in hx_values:
            print(f'\n\nhx={h}')
            my_hamil = hamil.toric_code_ladder_active_x(1, 1, h)
            itebd = tebd.iTEBD(
                my_hamil,
                physical_dim=phy_dim,
                virtual_dim=virtual_dim,
                unit_cells=unit_cells
            )
            mps = itebd.delta_manager(
                iteration=iteration,
                delta_steps=delta_steps,
                delta_start=delta_start,
                delta_end=delta_end
            )

            # MPS Bonds Energy
            mps_bonds_energy = itebd.mps_bonds_energy(mps, my_hamil)

            # Magnetization
            mag_profile = itebd.expectation_all_sites_mag(mps, 'xz')

            # Save MPS
            self.save_mps(mps)

            # Save Data
            df = pd.DataFrame([
                [phy_dim] +
                [virtual_dim] +
                [unit_cells] +
                [h] +
                mps_bonds_energy +
                [sum(mps_bonds_energy)] +
                mag_profile['x'] +
                mag_profile['z'] +
                [mag_profile['mean_x']] +
                [mag_profile['mean_z']] +
                [mag_profile['mag_value']]
            ])

            df.to_csv(self.physical_data, mode='a', header=False, index=False)

        df = pd.read_csv(self.physical_data, names=[
            'phy_dim',
            'vir_dim',
            'unit_cells',
            'hx',
            'energy_bond_1',
            'energy_bond_2',
            'energy',
            'mag_x_1', 'mag_x_2', 'mag_x_3', 'mag_x_4', 'mag_x_5', 'mag_x_6',
            'mag_z_1', 'mag_z_2', 'mag_z_3', 'mag_z_4', 'mag_z_5', 'mag_z_6',
            'mag_mean_x',
            'mag_mean_z',
            'mag_value',
        ])
        df.to_csv(self.physical_data, header=True, index=False)

    def data_sampling_2uc(
            self,
            matrix_type: str = 'pauli',
            virtual_dim: int = 2,
            unit_cells: int = 2,
            hx_min: float = 0.0,
            hx_max: float = 1.4,
            hx_steps: float = 0.05,
            iteration: int = 3000,
            delta_start: int = 0.1,
            delta_end: int = 0.01,
            delta_steps: int = 3,
    ):
        phy_dim = 8
        hamil = hml.Hamiltonian(matrix_type=matrix_type)
        hx_values = np.linspace(hx_min, hx_max, num=int((hx_max - hx_min) / hx_steps))

        for h in hx_values:
            print(f'\n\nhx={h}')
            my_hamil = hamil.toric_code_ladder_active_x(1, 1, h)
            itebd = tebd.iTEBD(
                my_hamil,
                physical_dim=phy_dim,
                virtual_dim=virtual_dim,
                unit_cells=unit_cells
            )
            mps = itebd.delta_manager(
                iteration=iteration,
                delta_steps=delta_steps,
                delta_start=delta_start,
                delta_end=delta_end
            )

            # MPS Bonds Energy
            mps_bonds_energy = itebd.mps_bonds_energy(mps, my_hamil)

            # Magnetization
            mag_profile = itebd.expectation_all_sites_mag(mps, 'xz')

            # Save MPS
            self.save_mps(mps)

            # Save Data
            df = pd.DataFrame([
                [phy_dim] +
                [virtual_dim] +
                [unit_cells] +
                [h] +
                mps_bonds_energy +
                [sum(mps_bonds_energy)] +
                mag_profile['x'] +
                mag_profile['z'] +
                [mag_profile['mean_x']] +
                [mag_profile['mean_z']] +
                [mag_profile['mag_value']]
            ])

            df.to_csv(self.physical_data, mode='a', header=False, index=False)

        df = pd.read_csv(self.physical_data, names=[
            'phy_dim',
            'vir_dim',
            'unit_cells',
            'hx',
            'energy_bond_1',
            'energy_bond_2',
            'energy_bond_3',
            'energy_bond_4',
            'energy',
            'mag_x_1', 'mag_x_2', 'mag_x_3', 'mag_x_4', 'mag_x_5', 'mag_x_6',
            'mag_x_7', 'mag_x_8', 'mag_x_9', 'mag_x_10', 'mag_x_11', 'mag_x_12',
            'mag_z_1', 'mag_z_2', 'mag_z_3', 'mag_z_4', 'mag_z_5', 'mag_z_6',
            'mag_z_7', 'mag_z_8', 'mag_z_9', 'mag_z_10', 'mag_z_11', 'mag_z_12',
            'mag_mean_x',
            'mag_mean_z',
            'mag_value',
        ])
        df.to_csv(self.physical_data, header=True, index=False)
