import numpy as np
import torch
from Make_dataset import Read_data, Make_dataset
from torch.utils.data import DataLoader
from config import DefaultConfig

def debug_data_dimensions():
    # 加载配置
    config = DefaultConfig()
    
    # 加载数据集
    reader = Read_data(config.train_data_root, config.valid_ratio)
    train_set, valid_set = reader.read_file()
    
    print(f"Train set shape: {train_set.shape}")
    print(f"Valid set shape: {valid_set.shape}")
    
    # 制作Loader
    TrainSet = Make_dataset(train_set)
    train_loader = DataLoader(dataset=TrainSet, batch_size=config.batch_size, shuffle=True)
    
    # 检查一个批次的数据
    for idx, noise in enumerate(train_loader):
        print(f"Batch {idx} - noise shape: {noise.shape}")
        noise = np.squeeze(noise.numpy())
        print(f"After squeeze - noise shape: {noise.shape}")
        
        if len(noise.shape) == 3:
            # 如果是3D张量 (batch, 1, features)
            spectra_num, channels, spec = noise.shape
            print(f"3D tensor - spectra_num: {spectra_num}, channels: {channels}, spec: {spec}")
        elif len(noise.shape) == 2:
            # 如果是2D张量 (batch, features)
            spectra_num, spec = noise.shape
            print(f"2D tensor - spectra_num: {spectra_num}, spec: {spec}")
        else:
            print(f"Unexpected shape: {noise.shape}")
            
        # 检查第一个样本的维度
        if len(noise.shape) == 3:
            sample = noise[0]  # (1, features)
            print(f"First sample shape: {sample.shape}")
        elif len(noise.shape) == 2:
            sample = noise[0]  # (features,)
            print(f"First sample shape: {sample.shape}")
            
        break  # 只检查第一个批次

if __name__ == '__main__':
    debug_data_dimensions()