import cv2
import numpy as np
from .gesture_recognizer_base import GestureRecognizerBase

class CustomRecognizer(GestureRecognizerBase):
    """自定义的手势识别器"""
    
    def __init__(self):
        super().__init__()
        self.background_subtractor = None
        
    def initialize(self):
        """初始化自定义识别器"""
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
    
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, str]:
        if not self.is_running:
            return frame, "未启动"
            
        # 背景分割
        fg_mask = self.background_subtractor.apply(frame)
        
        # 形态学操作
        kernel = np.ones((5,5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(
            fg_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        gesture_result = "无手势"
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            if area > 5000:  # 面积阈值
                # 绘制轮廓
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                
                # 计算凸包
                hull = cv2.convexHull(contour)
                cv2.drawContours(frame, [hull], -1, (0, 0, 255), 2)
                
                # 简单的手势判断（示例）
                if len(hull) > 10:
                    gesture_result = "检测到手部"
        
        return frame, gesture_result 