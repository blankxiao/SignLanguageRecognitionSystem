from abc import ABC, abstractmethod
import numpy as np
import cv2

class GestureRecognizerBase(ABC):
    """手势识别器基类"""
    
    def __init__(self):
        self.is_running = False
    
    @abstractmethod
    def initialize(self):
        """初始化识别器"""
        pass
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, str]:
        """处理单帧图像
        
        Args:
            frame: 输入的图像帧
            
        Returns:
            tuple: (处理后的图像帧, 识别结果)
        """
        pass
    
    def start(self):
        """启动识别器"""
        self.is_running = True
    
    def stop(self):
        """停止识别器"""
        self.is_running = False
    
    def release(self):
        """释放资源"""
        self.stop() 