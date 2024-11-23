"""
@Author: blankxiao
@file: utils.py
@Created: 2024-11-23
@Desc: 工具函数模块
"""

from typing import Optional, Tuple, Any
import cv2
import numpy as np
import torch
from pathlib import Path

def ensure_dir(dir_path: Path) -> None:
    """确保目录存在，如果不存在则创建"""
    dir_path.mkdir(parents=True, exist_ok=True)

def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    加载图像
    
    Args:
        image_path: 图像路径
        
    Returns:
        Optional[np.ndarray]: 加载的图像，如果加载失败返回None
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法加载图像: {image_path}")
        return None
    return img

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    预处理图像
    
    Args:
        image: 输入图像
        target_size: 目标大小
        
    Returns:
        np.ndarray: 预处理后的图像
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 调整大小
    image = cv2.resize(image, target_size)
    
    # 归一化
    image = image.astype(np.float32) / 255.0
    
    return image 