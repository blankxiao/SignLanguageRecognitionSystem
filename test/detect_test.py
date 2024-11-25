"""
@Author: blankxiao
@file: detect_test.py
@Created: 2024-11-23
@Desc: 测试手部检测的各个处理步骤
"""

import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def test_hand_detection():
    """测试手部检测的各个步骤"""
    try:
        # 读取测试图片
        img_path = Path("./testimg/test.jpg")
        if not img_path.exists():
            logger.error(f"测试图片不存在: {img_path}")
            return
        
        # 读取并显示原始图片
        original = cv2.imread(str(img_path))
        cv2.imshow("1. Original Image", original)
        cv2.imwrite("testimg/1_original.jpg", original)
        
        # 转换到YCrCb颜色空间
        ycrcb = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)
        cv2.imshow("2. YCrCb Color Space", ycrcb)
        cv2.imwrite("testimg/2_ycrcb.jpg", ycrcb)
        
        # 应用肤色掩码
        # YCrCb空间中的肤色范围
        min_YCrCb = np.array([0, 133, 77], np.uint8)
        max_YCrCb = np.array([255, 173, 127], np.uint8)
        
        # 创建掩码
        skin_mask = cv2.inRange(ycrcb, min_YCrCb, max_YCrCb)
        cv2.imshow("3. Skin Mask", skin_mask)
        cv2.imwrite("testimg/3_skin_mask.jpg", skin_mask)
        
        # 形态学处理
        # 创建结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # 闭运算：先膨胀后腐蚀，填充小洞
        closed = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("4. After Closing", closed)
        cv2.imwrite("testimg/4_after_closing.jpg", closed)
        
        # 开运算：先腐蚀后膨胀，去除小斑点
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        cv2.imshow("5. After Opening", opened)
        cv2.imwrite("testimg/5_after_opening.jpg", opened)
        
        # 再次膨胀，扩大皮肤区域
        dilated = cv2.dilate(opened, kernel, iterations=2)
        cv2.imshow("6. After Dilation", dilated)
        cv2.imwrite("testimg/6_after_dilation.jpg", dilated)
        
        # 找到轮廓
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 在原图上绘制轮廓
        result = original.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        cv2.imshow("7. Final Result", result)
        cv2.imwrite("testimg/7_final_result.jpg", result)
        
        logger.info("手部检测测试完成，结果已保存到testimg目录")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.error(f"测试过程出错: {str(e)}")

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_hand_detection()
