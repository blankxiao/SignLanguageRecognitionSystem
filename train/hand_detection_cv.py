"""
@Author: blankxiao
@file: hand_detection_cv.py
@Created: 2024-11-23
@Desc: 使用传统计算机视觉方法实现手部检测
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np

class HandDetector:
    def __init__(self) -> None:
        # 肤色检测的YCrCb阈值
        self.min_YCrCb: np.ndarray = np.array([0, 133, 77], np.uint8)
        self.max_YCrCb: np.ndarray = np.array([255, 173, 127], np.uint8)
        
        # 形态学操作的核
        self.kernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # 最小和最大手部面积（用于过滤区域）
        self.min_hand_area: int = 5000
        self.max_hand_area: int = 50000
    
    def detect_skin(self, image: np.ndarray) -> np.ndarray:
        """
        使用YCrCb颜色空间检测皮肤
        
        Args:
            image: BGR格式的输入图像
            
        Returns:
            二值化的皮肤掩码
        """
        # 转换到YCrCb颜色空间
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # 创建皮肤掩码
        skin_mask = cv2.inRange(ycrcb_image, self.min_YCrCb, self.max_YCrCb)
        
        # 形态学操作改善掩码
        skin_mask = cv2.erode(skin_mask, self.kernel, iterations=2)
        skin_mask = cv2.dilate(skin_mask, self.kernel, iterations=2)
        
        return skin_mask
    
    def find_hand_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        找到所有可能的手部轮廓
        
        Args:
            mask: 二值化掩码图像
            
        Returns:
            符合面积条件的轮廓列表
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤并保留合适大小的轮廓
        hand_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_hand_area <= area <= self.max_hand_area:
                hand_contours.append(contour)
        
        return hand_contours
    
    def detect_hands(self, image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        检测多个手部
        
        Args:
            image: BGR格式的输入图像
            
        Returns:
            Tuple[标注后的图像, 手部ROI列表]
        """
        # 复制输入图像
        result = image.copy()
        
        # 检测皮肤
        skin_mask = self.detect_skin(image)
        
        # 找到所有手部轮廓
        hand_contours = self.find_hand_contours(skin_mask)
        
        # 存储所有手部ROI
        hand_rois: List[np.ndarray] = []
        
        # 处理每个检测到的手部区域
        for i, contour in enumerate(hand_contours):
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 添加边距
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            # 提取ROI
            hand_roi = image[y1:y2, x1:x2]
            if hand_roi.size > 0:
                hand_rois.append(hand_roi)
                
                # 在原图上绘制边界框和标签
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(result, f"Hand {i+1}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result, hand_rois

def main() -> None:
    """主函数，用于测试手部检测器"""
    # 创建检测器
    detector = HandDetector()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 检测手部并获取结果
        result, hand_rois = detector.detect_hands(frame)
        
        # 显示结果（只显示一个窗口）
        cv2.imshow('Hand Detection', result)
        
        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 清理
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 