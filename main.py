import pandas as pd
import hamiltonian as hml
import itebd as tbd
import diagram as dgm
import numpy as np
from ncon import ncon
import matplotlib.pyplot as plt


def plot_energy(file_name: str):
    data = pd.read_csv(f'mps_data/{file_name}', header=None, index_col=False)

    plt.plot(data.iloc[:, :])
    plt.title(f'Energy Plot')
    plt.ylabel('<E>')
    plt.xlabel('Iteration')
    plt.show()


hamil = hml.Hamiltonian()
# tbd.iTEBD(hamil.toric_code_ribbon_II(1, 1), 4, 2).iTEBD_trotter_manager(60000, 6)
tbd.iTEBD(hamil.toric_code_ribbon(1, 1), 8, 16).iTEBD_trotter_manager(60000, 6)

# plot_energy('1693132560_history_energy.csv')
# plot_energy('1693134830_history_energy.csv')

# dgm.Diagram(13).all_plot()
