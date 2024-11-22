"""
@Author: blankxiao
@file: dataset.py
@Created: 2024-11-22 22:56
@Desc: 数据集处理
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # 标签映射关系
        self.label_map = {0: 9, 1: 0, 2: 7, 3: 6, 4: 1, 5: 8, 6: 4, 7: 3, 8: 2, 9: 5}
        
        # 直接加载.npy文件
        self.data = np.load(f"{data_dir}/X.npy")  # shape: (2062, 64, 64)
        labels = np.load(f"{data_dir}/Y.npy")     # shape: (2062, 10)
        
        # 将one-hot编码转换为单个数字标签
        self.labels = np.argmax(labels, axis=1)
        
        # 应用标签映射
        self.labels = np.vectorize(self.label_map.get)(self.labels)
        
        # 打印一些数据统计信息用于调试
        print(f"数据集大小: {len(self.data)}")
        print(f"标签分布: {np.unique(self.labels, return_counts=True)}")
        print(f"图像值范围: [{self.data.min()}, {self.data.max()}]")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        # 确保数据是正确的形状 (1, 64, 64)
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]
            
        # 转换为torch tensor并归一化到[0,1]
        image = torch.from_numpy(image).float() / 255.0
        # 将标签转换为LongTensor类型
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label 