from typing import Tuple
import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, d_type: str = "train", normalize: bool = True, bounds: Tuple[int] = (0, 1),
                future_steps: int= 1, sequence_length: int = 1, precision: np.dtype = np.float32):
        super(Dataset, self).__init__()

        # TODO