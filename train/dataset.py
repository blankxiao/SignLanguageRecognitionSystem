"""
@Author: blankxiao
@file: dataset.py
@Created: 2024-11-22 22:56
@Desc: 数据集处理
"""

from typing import Tuple, Dict, Any
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir: str, transform: transforms.Compose = None) -> None:
        # 标签映射关系
        self.label_map: Dict[int, int] = {
            0: 9, 1: 0, 2: 7, 3: 6, 4: 1, 
            5: 8, 6: 4, 7: 3, 8: 2, 9: 5
        }
        
        # 加载数据
        self.data: np.ndarray = np.load(f"{data_dir}/X.npy")  # shape: (2062, 64, 64)
        labels: np.ndarray = np.load(f"{data_dir}/Y.npy")     # shape: (2062, 10)
        
        # 将one-hot编码转换为单个数字标签
        self.labels: np.ndarray = np.argmax(labels, axis=1)
        
        # 应用标签映射
        self.labels = np.vectorize(self.label_map.get)(self.labels)
        
        # 数据增强
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15),  # 旋转角度
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # 平移范围
                scale=(0.9, 1.1),      # 缩放范围
            ),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 添加标准化
        ]) if transform is None else transform
        
        # 打印数据统计信息
        print(f"数据集大小: {len(self.data)}")
        print(f"标签分布: {np.unique(self.labels, return_counts=True)}")
        print(f"图像值范围: [{self.data.min()}, {self.data.max()}]")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 获取图像和标签
        image = self.data[idx]
        label = self.labels[idx]
        
        # 确保图像是2D的
        if len(image.shape) == 3:
            image = image.squeeze()
        
        # 应用数据增强
        if self.transform:
            image = self.transform(image)
        
        # 将标签转换为tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label 