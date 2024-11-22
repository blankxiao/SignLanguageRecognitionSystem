import cv2
import numpy as np
import mediapipe as mp
from .gesture_recognizer_base import GestureRecognizerBase

class MediaPipeRecognizer(GestureRecognizerBase):
    """基于MediaPipe的手势识别器"""
    
    def __init__(self):
        super().__init__()
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = None
        
    def initialize(self):
        """初始化MediaPipe手势识别器"""
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, str]:
        if not self.is_running:
            return frame, "未启动"
            
        # 转换颜色空间
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        # 处理识别结果
        gesture_result = "无手势"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # 简单的手势判断逻辑（示例）
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                
                if thumb_tip.y < index_tip.y:
                    gesture_result = "竖起大拇指"
                else:
                    gesture_result = "其他手势"
        
        return frame, gesture_result
    
    def release(self):
        """释放资源"""
        super().release()
        if self.hands:
            self.hands.close() 