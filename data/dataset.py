from typing import Tuple
import numpy as np
import scipy.io
from sklearn.preprocessing import MinMaxScaler
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, d_type: str = "train", normalize: bool = True, bounds: Tuple[int] = (0, 1),
                future_steps: int= 1, sequence_length: int = 1, precision: np.dtype = np.float32):
        super(Dataset, self).__init__()

        self._precision = precision
        self._seq = sequence_length
        self._f_seq = future_steps
        
        # load the dataset specified
        self._file = f"./data/{d_type}.mat"
        self._mat = scipy.io.loadmat(self._file).get("X")
        self._mat = self._mat.astype(self._precision)

        # normalize the dataset between values of 0 to 1
        self._scaler = [None for _ in range(self.sample_size)]
        if normalize:
            for i, sc in enumerate(self._scaler):
                if sc is None:
                    sc = MinMaxScaler(feature_range=bounds)
                    self._scaler[i] = sc
                
                sc.fit(self._mat[:, :, i])
                self._mat[:, :, i] = sc.transform(self._mat[:, :, i])
    
    def print_row(self, row_index, city_index = 0) -> None:
        headers = ["Wind speed", "Direction", "Temperature", "Dew point", "Air pressure"]
        row = self._mat[row_index, city_index, :]
        for i, header in enumerate(headers):
            print(header, row[i])

    @property
    def sample_size(self) -> int:
        return 5
    
    def scale_back(self, data):
        data = np.array(data, dtype=self._precision)
        return self._scaler.inverse_transform(data)

    def __len__(self):
        return max(1, len(self._mat) - self._f_seq - self._seq)

    def __getitem__(self, index):
        X = self._mat[index:self._seq + index, :, :]
        y = self._mat[self._seq + index:self._seq + index + self._f_seq, -1, 2]
        return X, y