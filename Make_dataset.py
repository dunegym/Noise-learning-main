"""
Author: HaoHe
Email: hehao@stu.xmu.edu.cn
Date: 2020-09-13
"""

import numpy as np
import os
import torch
import scipy.io as sio
import h5py
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class Read_data:
    def __init__(self, path, valid_ratio=20):
        self.path = path
        self.valid_ratio = valid_ratio

    def _load_mat_file(self, file_path):
        """
        Load .mat file, automatically detecting v7.3 (HDF5) or older format
        """
        try:
            # First try to load as regular .mat file
            tmp = sio.loadmat(file_path)
            # Extract noise data
            for key in tmp.keys():
                if key == 'noise':
                    data = np.array(tmp[key])
                    return data
        except NotImplementedError:
            # If that fails, try loading as HDF5 file (v7.3)
            with h5py.File(file_path, 'r') as f:
                # Extract noise data
                if 'noise' in f:
                    data = np.array(f['noise'])
                    # Note: HDF5 data may have different orientation
                    return data
                else:
                    # Try to find any dataset that looks like noise data
                    for key in f.keys():
                        if key == 'noise' or 'noise' in key.lower():
                            data = np.array(f[key])
                            return data
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            raise
        return None

    def read_file(self):
        """
        Args:
            valid_ratio: the ratio of the validation sets
            path: the absolute path of the source file
        """

        filenames = os.listdir(self.path)
        train_data, valid_data = [], []
        for filename in filenames:
            # 分别存储单个文件中的训练数据和验证数据及标签
            tmp_train_data, tmp_valid_data = [], []
            if os.path.splitext(filename)[1] == '.mat':
                # 绝对路径+文件名
                name = self.path + '/' + filename
                try:
                    data = self._load_mat_file(name)
                    if data is not None:
                        data = data.T
                        num, spec = data.shape
                        valid_num = int(np.ceil(self.valid_ratio / 100 * num))
                        tmp_valid_data, tmp_train_data = data[0:valid_num], data[valid_num:]
                        train_data.append(tmp_train_data)
                        valid_data.append(tmp_valid_data)
                except Exception as e:
                    print(f"Error processing file {name}: {e}")
                    continue
                    
        if not train_data or not valid_data:
            raise ValueError("No valid data found in the specified path")
            
        train_data_tmp = np.array(train_data[0])
        valid_data_tmp = np.array(valid_data[0])

        # list 拼接成numpy数组
        if len(train_data) > 1:
            for i in range(1, len(train_data)):
                train_data_tmp = np.concatenate((train_data_tmp, train_data[i]), axis=0)
                valid_data_tmp = np.concatenate((valid_data_tmp, valid_data[i]), axis=0)

        # Get the spec dimension from the data
        _, _, spec = train_data_tmp.shape if len(train_data_tmp.shape) == 3 else (train_data_tmp.shape[0], 1, train_data_tmp.shape[-1])
        if len(train_data_tmp.shape) == 2:
            train_data_tmp = train_data_tmp.reshape((-1, 1, spec))
            valid_data_tmp = valid_data_tmp.reshape((-1, 1, spec))
        else:
            # Ensure correct shape
            train_dataset = train_data_tmp
            valid_dataset = valid_data_tmp

        return train_data_tmp, valid_data_tmp


# 制作数据集
class Make_dataset(Dataset):
    def __init__(self, data):
        super(Make_dataset, self).__init__()
        self.data = data

    def __getitem__(self, index):
        feature = self.data[index]
        return feature

    def __len__(self):
        return np.size(self.data, 0)


if __name__ == '__main__':
    path = r'E:\PAPER\paper writing\Noise learning\Simulate datasets'
    reader = Read_data(path, 20)
    train_set, _ = reader.read_file()
    print(train_set.shape)
    x = np.mean(train_set[0], axis=0)
    print(x.shape)
    plt.plot(x.ravel())
    # test dataloader
    myset = Make_dataset(train_set)
    train_loader = DataLoader(dataset=myset, batch_size=100)
    print(len(train_loader))
    for batch_idx, raw in enumerate(train_loader):
        if batch_idx > 1:
            print(raw.shape)
            x = np.mean(raw.numpy(), axis=0)
            print(x.shape)
            plt.plot(x)
            plt.show()
            break
