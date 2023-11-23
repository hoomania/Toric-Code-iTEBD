from iTEBD import hamiltonian as hml, itebd as tebd
from storage import export as datalog
import numpy as np
import os
import pandas as pd
import time


def data_sampling(
        matrix_type: str = 'pauli',
        virtual_dim: int = 2,
        unit_cells: int = 1,
        iteration: int = 3000,
        delta_start: int = 0.1,
        delta_end: int = 0.01,
        delta_steps: int = 3,
):
    phy_dim = 8
    hamil = hml.Hamiltonian(matrix_type=matrix_type)
    hx_values = [np.round(i * 0.05, 2) for i in range(17, 19)]
    hz_values = [np.round(i * 0.05, 2) for i in range(10, 11)]

    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/logs/' + time.strftime("%Y_%m_%d_%H_%M_%S_UTC",
                                                                                      time.gmtime())
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        os.makedirs(dir_path + '/mps_pickle')
    elif not os.path.exists(dir_path + '/mps_pickle'):
        os.makedirs(dir_path + '/mps_pickle')

    export_data = datalog.Log(dir_path)
    physical_data_path = f'{dir_path}/phy_data.csv'
    mps_profile_data_path = f'{dir_path}/mps_profile_data.csv'

    for hx in hx_values:
        for hz in hz_values:
            print('\n\n' + time.strftime("%H:%M:%S (UTC)", time.gmtime()))
            print(f'hx={hx} hz={hz}')
            my_hamil = hamil.toric_code_ladder_active_xz(1, 1, hx=hx, hz=hz)
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

            # Save MPS
            export_data.save_mps_pickle(
                mps,
                f'{dir_path}/mps_pickle/hx-{int(np.round(hx * 100))}_hz{int(np.round(hz * 100))}'
            )
            df = pd.DataFrame([
                [phy_dim] + [virtual_dim] + [unit_cells] + [hx] + [hz] +
                [f'hx-{int(np.round(hx * 100))}_hz{int(np.round(hz * 100))}.pkl']
            ])
            df.to_csv(mps_profile_data_path, mode='a', header=False, index=False)

            # MPS Bonds Energy
            mps_bonds_energy = itebd.mps_bonds_energy(mps, my_hamil)

            # Magnetization
            mag_profile = itebd.expectation_all_sites_mag(mps, 'xz')

            # Save Data
            df = pd.DataFrame([
                [phy_dim] +
                [virtual_dim] +
                [unit_cells] +
                [hx] +
                [hz] +
                mps_bonds_energy +
                [sum(mps_bonds_energy)] +
                mag_profile['x'] +
                mag_profile['z'] +
                [mag_profile['mean_x']] +
                [mag_profile['mean_z']] +
                [mag_profile['mag_value']]
            ])

            df.to_csv(physical_data_path, mode='a', header=False, index=False)

    df = pd.read_csv(mps_profile_data_path, names=[
        'phy_dim',
        'vir_dim',
        'unit_cells',
        'hx',
        'hz',
        'file_name',
    ])
    df.to_csv(mps_profile_data_path, header=True, index=False)

    df = pd.read_csv(physical_data_path, names=[
        'phy_dim',
        'vir_dim',
        'unit_cells',
        'hx',
        'hz',
        'energy_bond_1', 'energy_bond_2', 'energy',
        'mag_x_1', 'mag_x_2', 'mag_x_3', 'mag_x_4', 'mag_x_5', 'mag_x_6',
        'mag_z_1', 'mag_z_2', 'mag_z_3', 'mag_z_4', 'mag_z_5', 'mag_z_6',
        'mag_mean_x', 'mag_mean_z', 'mag_value',
    ])
    df.to_csv(physical_data_path, header=True, index=False)


data_sampling(
    virtual_dim=32
)
