from iTEBD import hamiltonian as hml, itebd as tebd
import numpy as np
import os
import pandas as pd
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
time_prefix = time.strftime("%Y_%m_%d_%H_%M_%S_UTC", time.gmtime())
physical_data = f'{dir_path}/logs/{time_prefix}_phy_data_example.csv'
mps_data = f'{dir_path}/logs/{time_prefix}_mps_data_example.csv'


def save_mps(
        mps: list,
        phy_dim: int,
        vir_dim: int,
) -> None:
    reshape_val = 2 * ((vir_dim * phy_dim * vir_dim) + vir_dim)

    mps_row = []
    for item in mps:
        if len(item.shape) == 3:
            mps_row = np.hstack([mps_row, np.reshape(item, (-1,))])
        else:
            mps_row = np.hstack([mps_row, np.diag(item)])

    df = pd.DataFrame(mps_row.reshape(1, reshape_val))
    df.to_csv(mps_data, mode='a', header=False, index=False)


def data_sampling(
        matrix_type: str = 'pauli',
        virtual_dim: int = 2,
        unit_cells: int = 1,
        hx_min: float = 0.0,
        hx_max: float = 3.0,
        hx_steps: float = 0.05,
        iteration: int = 3000,
        delta_start: int = 0.1,
        delta_end: int = 0.01,
        delta_steps: int = 3,
):
    phy_dim = 2
    hamil = hml.Hamiltonian(matrix_type=matrix_type)
    hx_values = np.linspace(hx_min, hx_max, num=int((hx_max - hx_min) / hx_steps))

    for h in hx_values:
        print(f'\n\nhx={h}')
        my_hamil = hamil.transverse_field_ising(1, h)
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
        save_mps(mps, phy_dim, virtual_dim)

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

        df.to_csv(physical_data, mode='a', header=False, index=False)

    df = pd.read_csv(physical_data, names=[
        'phy_dim',
        'vir_dim',
        'unit_cells',
        'hx',
        'energy_bond_1',
        'energy_bond_2',
        'energy',
        'mag_x_1',
        'mag_x_2',
        'mag_z_1',
        'mag_z_2',
        'mag_mean_x',
        'mag_mean_z',
        'mag_value',
    ])
    df.to_csv(physical_data, header=True, index=False)


data_sampling(hx_max=1.0, hx_steps=0.05)
