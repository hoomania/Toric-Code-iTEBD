import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Diagram:
    def __init__(self, index_row: int):
        profile = pd.read_csv('mps_data/data_profile.csv', header=None, index_col=False)

        self.file_capsule = profile.loc[index_row, 7]
        self.file_element = profile.loc[index_row, 8]
        self.file_energy = profile.loc[index_row, 9]
        self.fig_title = profile.loc[index_row, 1]
        self.index_row = index_row

    def energy_plot(self, ax):
        data = pd.read_csv(f'mps_data/{self.file_energy}', header=None, index_col=False)

        ax.plot(data.iloc[:, -1])
        ax.set_title(f'energy (converged to: {data.iloc[-1, -1]})')
        ax.set_ylabel('<E>')
        ax.set_xlabel('Iteration')

    def capsule_plot(self, ax, color_map: str = 'hot'):
        data = pd.read_csv(f'mps_data/{self.file_capsule}')
        data = data.iloc[:, :].transpose()
        data = self.normalize(data)

        im = ax.imshow(
            data.iloc[:, :],
            cmap=color_map,
            aspect="auto"
        )
        plt.colorbar(im, orientation='horizontal')
        ax.set_title(f'MPS contract evolution (left to right)')
        ax.set_xlabel('iteration')
        # ax.tick_params(
        #     axis='x',
        #     which='both',
        #     bottom=False,
        #     top=False,
        #     labelbottom=False
        # )

    def element_plot(self, ax, color_map: str = 'hot'):
        data = pd.read_csv(f'mps_data/{self.file_element}')
        data = data.iloc[:, :].transpose()
        data = self.normalize(data)
        im = ax.imshow(
            data.iloc[:, :],
            cmap=color_map,
            aspect="auto"
            # extent=[0, data_length * 64, 0, data_height]
        )
        plt.colorbar(im, orientation='horizontal')
        ax.set_title(f'MPS element evolution (left to right)')
        ax.set_xlabel('iteration')
        # ax.tick_params(
        #     axis='x',
        #     which='both',
        #     bottom=False,
        #     top=False,
        #     labelbottom=False
        # )

    def first_last_step(self, ax, color_map: str = 'hot'):
        data = pd.read_csv(f'mps_data/{self.file_element}')
        min_ = np.min(data)
        max_ = np.max(data)

        data = [
            np.hstack([[min_, max_], data.iloc[-1, :]]),
            np.hstack([[min_, max_], data.iloc[0, :]]),
        ]
        data = (data - min_) / (max_ - min_)

        ax.grid(which='major', axis='y', linestyle='-', color='k', linewidth=1)

        # im = ax.imshow(data, cmap=color_map, aspect='auto')
        # plt.colorbar(im, orientation='horizontal')

        pcc = ax.pcolormesh(data, cmap=color_map, linewidth=0)
        plt.colorbar(pcc, orientation='horizontal')
        ax = plt.gca()
        ax.set_aspect('auto')

        # ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        ax.set_title('top: before variation / bottom: after variation')

        ax.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False
        )

        ax.tick_params(
            axis='y',
            which='both',
            left=False,
            right=False,
            labelleft=False
        )

    def normalize(self, data):
        data = data.iloc[1:, 1:]
        min_ = np.min(data)
        max_ = np.max(data)
        return (data - min_) / (max_ - min_)

    def all_plot(self) -> None:
        plt.figure(figsize=(8, 16))
        plt.suptitle(f'{self.fig_title} / row index: {self.index_row}')
        gs = gridspec.GridSpec(4, 1,
                               width_ratios=[1],
                               height_ratios=[3, 3, 3, 1],
                               hspace=0.4,
                               )

        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[1, 0])
        ax3 = plt.subplot(gs[2, 0])
        ax4 = plt.subplot(gs[3, 0])

        self.energy_plot(ax1)
        self.element_plot(ax2)
        self.capsule_plot(ax3)
        self.first_last_step(ax4)

        plt.show()

    def energy(self) -> None:
        plt.figure(figsize=(8, 16))
        fig, ax = plt.subplots()
        fig.suptitle(f'{self.fig_title} / row index: {self.index_row}')
        self.energy_plot(ax)

        plt.show()
