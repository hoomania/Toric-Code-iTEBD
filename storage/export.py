import numpy as np
import os
import pandas as pd
import time


class Log:
    def __init__(
            self,
            log_path: str
    ):
        time_prefix = time.strftime("%Y_%m_%d_%H_%M_%S_UTC", time.gmtime())
        self.file_name = f'{time_prefix}_mps_data.csv'
        self.log_path = log_path

    def save_mps(
            self,
            mps: list,
            file_name=None
    ) -> None:

        if file_name is not None:
            self.file_name = file_name

        mps_row = []
        for item in mps:
            if len(item.shape) == 3:
                mps_row = np.hstack([mps_row, np.reshape(item, (-1,))])
            else:
                mps_row = np.hstack([mps_row, np.diag(item)])

        df = pd.DataFrame(mps_row.reshape(1, len(mps_row)))
        df.to_csv(f'{self.log_path}/{self.file_name}', mode='a', header=False, index=False)
