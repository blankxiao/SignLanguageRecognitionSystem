"""
@Author: blankxiao
@file: mediapipe_recognizer.py
@Modified: 2024-11-23
@Desc: 使用MediaPipe的手势识别器，支持土耳其手语数字0-9的识别
"""

from typing import Optional, Tuple, List
import cv2
import mediapipe as mp
import numpy as np
from .gesture_recognizer_base import GestureRecognizerBase
import logging

logger = logging.getLogger(__name__)

class MediaPipeRecognizer(GestureRecognizerBase):
    def __init__(self) -> None:
        super().__init__()
        self.mp_hands = None
        self.hands = None
        self.mp_draw = None
        
        # 手指关键点索引
        # 顺序对应 拇指、食指、中指、无名指、小指
        self.finger_tips = [4, 8, 12, 16, 20]  # 指尖
        self.finger_mcp = [1, 5, 9, 13, 17]    # 掌指关节点
        self.finger_pips = [2, 6, 10, 14, 18]  # 近节指关节
        self.finger_dips = [3, 7, 11, 15, 19]  # 远节指关节
    
    def initialize(self) -> bool:
        try:
            # 获取mediapipe的工具
            self.mp_hands = mp.solutions.hands
            # 获取手部位置信息
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            # 处理帧的工具
            self.mp_draw = mp.solutions.drawing_utils
            return True
        except Exception as e:
            logger.error(f"MediaPipe初始化失败: {str(e)}")
            return False
    
    def _get_finger_states(self, landmarks) -> List[bool]:
        """获取所有手指的状态（弯曲/伸直）"""
        finger_states = []
        
        def calculate_collinearity(p1, p2, p3) -> float:
            """
            计算三个点的共线性，返回最小均方误差
            使用点到直线距离的方法计算
            """
            # 将landmarks点转换为numpy数组
            points = np.array([[p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y]])
            
            # 获取xy坐标
            x = points[:, 0]
            y = points[:, 1]
            
            # 最小二乘法拟合直线
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            
            # 点到拟合直线的距离平方和
            distances = (y - (m * x + c)) ** 2
            mse = np.mean(distances)
            
            return mse
        
        # 处理所有手指
        for idx in range(5):  # 从拇指到小指
            tip = landmarks[self.finger_tips[idx]]
            pip = landmarks[self.finger_pips[idx]]
            dip = landmarks[self.finger_dips[idx]]
            mcp = landmarks[self.finger_mcp[idx]]
            
            # 计算三个关节点的共线性
            if idx == 0:  # 拇指
                # 对于拇指，我们使用MCP、IP和指尖三个点
                mse = calculate_collinearity(mcp, pip, tip)
                # 大拇指如果根据指尖判断会出问题 因此这里直接忽略指尖的条件
                finger_extended = mse < 0.001
            else:  # 其他手指
                # 使用PIP、DIP和指尖三个点
                mse = calculate_collinearity(pip, dip, tip)
                # 其他手指伸直时还需要考虑y坐标的位置关系
                finger_extended = mse < 0.001 and tip.y < pip.y < mcp.y
            
            finger_states.append(finger_extended)
        
        return finger_states
    
    def _recognize_number(self, landmarks) -> Tuple[Optional[int], float]:
        """
        识别土耳其手语数字手势
        返回: (预测的数字, 置信度)
        """
        try:
            finger_states = self._get_finger_states(landmarks)

            # 定义手势规则
            if sum(finger_states) == 0:  # 0: 所有手指弯曲
                return 0, 0.9
            # 1: 仅食指伸直
            elif not finger_states[0] and finger_states[1] and not any(finger_states[2:]):
                return 1, 0.9
            # 3: 拇指、食指和中指伸直
            elif (finger_states[0] and finger_states[1] and finger_states[2] and 
                  not finger_states[3] and not finger_states[4]):
                return 3, 0.9
            # 2: 因为符合3就一定符合2 因此先判断3再判断2
            # 2: 食指和中指伸直
            elif not finger_states[0] and all(finger_states[1:3]) and not any(finger_states[3:]):
                return 2, 0.9
            # 5: 所有手指伸直，且手掌朝上
            elif (all(finger_states[1:]) and finger_states[0]):
                return 5, 0.9
            # 与3 2 同理
            # 4: 除拇指外都伸直
            elif not finger_states[0] and all(finger_states[1:]):
                return 4, 0.9
            # 6
            elif not finger_states[0] and all(finger_states[1:4]) and not finger_states[4]:
                return 6, 0.9
            # 7
            elif not finger_states[0] and finger_states[1] and finger_states[2] and not finger_states[3] and finger_states[4]:
                return 7, 0.9
            # 8
            elif not finger_states[0] and finger_states[1] and not finger_states[2] and finger_states[3] and finger_states[4]:
                return 8, 0.9
            # 9
            elif not finger_states[0] and not finger_states[1] and all(finger_states[2:]):
                return 9, 0.9
            
            return None, 0.0
            
        except Exception as e:
            logger.error(f"手势识别错误: {str(e)}")
            return None, 0.0
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[int]]:
        """
        输入原始帧 检测手部并分析手部的数字
        返回处理后的帧
        """
        try:
            # 转换颜色空间
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # mediapipe的结果
            results = self.hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                # 检测多个手
                for hand_landmarks in results.multi_hand_landmarks:
                    # 绘制手部关键点和连接线
                    self.mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    
                    # 识别手势数字
                    number, confidence = self._recognize_number(hand_landmarks.landmark)
                    return frame, number
            
            return frame, None
            
        except Exception as e:
            logger.error(f"处理帧时出错: {str(e)}")
            return frame, None
    
    def release(self) -> None:
        if self.hands:
            self.hands.close()
        self.mp_hands = None
        self.hands = None
        self.mp_draw = None