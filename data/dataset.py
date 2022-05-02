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
        self._file = "./data/data.mat"
        self._mat = scipy.io.loadmat(self._file).get("X")
        self._mat = self._mat.astype(self._precision)

        # the train set includes all values until the las 168 ones (1 week)
        if d_type == "train":
            self._mat = self._mat[:-168]
        else:
            self._mat = self._mat[-168:]

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
        return self._mat.shape[-1]
    
    def scale_back(self, data):
        temp_scaler = self._scaler[2]

        data = np.array(data, dtype=self._precision)
        dummy_data = np.zeros((len(data), 4), dtype=self._precision)
        dummy_data[:, 3] = [_ for _ in data]

        scaled_dummy_data = temp_scaler.inverse_transform(dummy_data)
        return scaled_dummy_data[:, 3]

    def __len__(self):
        return max(1, len(self._mat) - self._f_seq - self._seq)

    def __getitem__(self, index):
        X = self._mat[index:self._seq + index, :, :]
        # y is every value after the sequence until future steps, 
        # the temperature (at index 2) from the last city (at index -1)
        y = self._mat[self._seq + index:self._seq + index + self._f_seq, -1, 2]
        return X, y