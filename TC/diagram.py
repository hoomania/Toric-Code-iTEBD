import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d16_uc1_x_increase = pd.read_csv(
    '/home/hooman/Projects/python/toric_code_itedb/TC/logs/2023_10_31_06_15_20_UTC_phy_data_example.csv',
    header=0, usecols=['hx', 'mag_mean_x', 'mag_mean_z'], index_col=False)

# print(profile.iloc[1:, 3])
plt.plot(d16_uc1_x_increase['hx'], d16_uc1_x_increase['mag_mean_x'], label='mx')
plt.plot(d16_uc1_x_increase['hx'], d16_uc1_x_increase['mag_mean_z'], label='mz')
plt.title('Magnetization')
plt.xlabel('hx')
plt.ylabel('mean <mag>')
plt.legend()
plt.show()
