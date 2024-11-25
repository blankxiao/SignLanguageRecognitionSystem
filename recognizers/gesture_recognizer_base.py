"""
@Author: blankxiao
@file: gesture_recognizer_base.py
@Created: 2024-11-23
@Desc: 手势识别器基类，定义了所有手势识别器必须实现的接口
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any
import numpy as np

class GestureRecognizerBase(ABC):
    """手势识别器基类"""
    
    def __init__(self) -> None:
        """初始化基类"""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化识别器
        
        Returns:
            bool: 初始化是否成功
        """
        pass
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[int]]:
        """
        处理单帧图像
        
        Args:
            frame: 输入的图像帧
            
        Returns:
            Tuple[np.ndarray, Optional[int]]: (处理后的图像帧, 识别结果)
                - 处理后的图像帧：可能包含标注或可视化信息
                - 识别结果：如果识别成功返回手势类别，否则返回None
        """
        pass
    
    @abstractmethod
    def release(self) -> None:
        """释放资源"""
        pass 