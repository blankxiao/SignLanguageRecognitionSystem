"""
@Author: blankxiao
@file: mediapipe_recognizer.py
@Created: 2024-11-23
@Desc: 使用MediaPipe的手势识别器
"""

from typing import Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np
from .gesture_recognizer_base import GestureRecognizerBase

class MediaPipeRecognizer(GestureRecognizerBase):
    def __init__(self) -> None:
        super().__init__()
        self.mp_hands = None
        self.hands = None
        self.mp_draw = None
    
    def initialize(self) -> bool:
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7
            )
            self.mp_draw = mp.solutions.drawing_utils
            return True
        except Exception as e:
            print(f"MediaPipe初始化失败: {str(e)}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[int]]:
        if not self.is_running:
            return frame, None
            
        # 转换颜色空间
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        # 绘制结果
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
        
        return frame, None  # MediaPipe识别器不返回手势类别
    
    def release(self) -> None:
        if self.hands:
            self.hands.close()
        self.mp_hands = None
        self.hands = None
        self.mp_draw = None 