import segyio
import numpy as np
from pathlib import Path
from scipy.io import loadmat


def load_seismic_data(self, uploaded_directory):
    '''
    Load seismic data for experiments.
    '''
    if Path(uploaded_directory).suffix == '.npy':
        data = np.load(uploaded_directory)
    elif Path(uploaded_directory).suffix == '.mat':
        data = loadmat(uploaded_directory)
        keys = list(data.keys())
        keys.remove('__header__')
        keys.remove('__version__')
        keys.remove('__globals__')
        data = data[keys[0]]
    elif Path(uploaded_directory).suffix.lower() == '.sgy' or Path(uploaded_directory).suffix.lower() == '.segy':
        pass
    if data.ndim > 2:
        data = data[..., int(data.shape[-1] / 2)]
    else:  # only for data.npy
        data = data.T

    data = np.nan_to_num(data, nan=0)
    data = data / np.max(np.abs(data))

    # data direction
    data = np.nan_to_num(data, nan=0)
    if not np.all(data != 0, axis=0).any() and 'cube' not in uploaded_directory:
        data = data.T

    return data