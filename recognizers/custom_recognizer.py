import torch
import cv2
import numpy as np
from .gesture_recognizer_base import GestureRecognizerBase
from train.train import SignLanguageResNet

class CustomRecognizer(GestureRecognizerBase):
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def initialize(self):
        """初始化模型"""
        self.model = SignLanguageResNet()
        checkpoint = torch.load('train/models/best_model.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
    def process_frame(self, frame):
        """
        处理单帧图像 - 只进行预测，不修改原始帧
        Args:
            frame: BGR格式的图像
        Returns:
            gesture: 识别的手势数字
            frame: 原始帧
        """
        try:
            if self.model is None:
                return frame, None
                
            # 预处理图像用于预测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            normalized = resized.astype(np.float32) / 255.0
            tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # 模型推理
            with torch.no_grad():
                output = self.model(tensor)
                prediction = torch.argmax(output, dim=1).item()
            
            # 直接返回预测结果和原始帧
            return frame, prediction
            
        except Exception as e:
            print(f"Error in custom recognizer: {str(e)}")
            return frame, None
            
    def release(self):
        """释放资源"""
        self.model = None