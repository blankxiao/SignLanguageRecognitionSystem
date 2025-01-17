"""
@Author: blankxiao
@file: custom_recognizer.py
@Created: 2024-11-23
@Desc: 自定义手势识别器，使用传统CV方法
"""

from typing import Optional, Tuple, List
import cv2
import torch
import numpy as np
from pathlib import Path
from .gesture_recognizer_base import GestureRecognizerBase
from train.train import SignLanguageResNet
from detection.hand_detection_cv import HandDetector
from config import MODEL_CONFIG
import logging
import os

logger = logging.getLogger(__name__)

class CustomRecognizer(GestureRecognizerBase):
    def __init__(self, model_path: Optional[str] = None) -> None:
        super().__init__()
        self.initialized: bool = False
        # 手部识别
        self.hand_detector: Optional[HandDetector] = None
        # 模型
        self.model: Optional[SignLanguageResNet] = None
        self.device: Optional[torch.device] = None
        # 模型路径
        self.model_path = model_path or str(MODEL_CONFIG['model_save_dir'] / MODEL_CONFIG['model_name'])
    
    def initialize(self) -> bool:
        """初始化识别器"""
        try:
            # 初始化手部检测器
            self.hand_detector = HandDetector()
            
            # 初始化手势识别模型
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = SignLanguageResNet()
            self.load_model()
            self.model.eval()
            
            logger.info(f"自定义识别器初始化完成，使用模型: {self.model_path}")
            logger.info(f"运行设备: {self.device}")
            
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"自定义识别器初始化失败: {str(e)}")
            return False
    
    def load_model(self) -> None:
        """加载训练好的模型"""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
                
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if self.model is not None:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                
                # 打印模型信息
                if 'val_acc' in checkpoint:
                    logger.info(f"模型验证准确率: {checkpoint['val_acc']:.2f}%")
                if 'metrics' in checkpoint:
                    logger.info(f"模型F1分数: {np.mean(checkpoint['metrics']['f1']):.4f}")
        except Exception as e:
            raise Exception(f"加载模型失败: {str(e)}")
    
    def preprocess_hand_roi(self, hand_roi: np.ndarray) -> torch.Tensor:
        """
        预处理手部ROI区域
        
        Args:
            hand_roi: 手部区域的图像数组
            
        Returns:
            处理后的tensor
        """
        # 转换为灰度图
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        
        # 加自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # 调整大小到64x64
        resized = cv2.resize(gray, (64, 64))
        
        # 增强对比度
        normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
        
        # 归一化到[-1,1]范围
        normalized = (normalized.astype(np.float32) - 127.5) / 127.5
        
        # 转换为tensor
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        return tensor
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[int]]:
        """
        处理单帧图像
        
        Args:
            frame: 输入图像帧
            
        Returns:
            Tuple[处理后的图像帧, 预测的手势类别(如果检测到)]
        """
        if not self.initialized:
            raise RuntimeError("识别器未初始化")
        
        try:
            # 检测手部
            result_frame, hand_rois = self.hand_detector.detect_hands(frame)
            
            if not hand_rois:
                # 没有检测到手
                return result_frame, None
            
            # 对每个检测到的手部区域进行手势识别
            predictions: List[Tuple[int, float]] = []
            for hand_roi in hand_rois:
                tensor = self.preprocess_hand_roi(hand_roi)
                if self.device is not None:
                    tensor = tensor.to(self.device)
                
                if self.model is not None:
                    with torch.no_grad():
                        output = self.model(tensor)
                        probabilities = torch.softmax(output, dim=1)
                        prediction = torch.argmax(output, dim=1).item()
                        confidence = probabilities.max().item()
                        
                        predictions.append((prediction, confidence))
            
            if predictions:
                # 选择最大可能的预测
                best_prediction = max(predictions, key=lambda x: x[1])
                return result_frame, best_prediction[0]

            # 预测出错
            return result_frame, None
            
        except Exception as e:
            logger.error(f"处理帧时出错: {str(e)}")
            return frame, None
    
    def release(self) -> None:
        """释放资源"""
        self.initialized = False
        self.model = None
        self.hand_detector = None