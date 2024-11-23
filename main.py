"""
@Author: blankxiao
@file: main.py
@Created: 2024-11-23
@Desc: 主程序入口
"""

from typing import Optional, Any
import logging
import sys
from PyQt6.QtWidgets import QApplication
from ui.recognition_window import RecognitionWindow
from recognizers.custom_recognizer import CustomRecognizer
from recognizers.mediapipe_recognizer import MediaPipeRecognizer

def setup_logging() -> None:
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main() -> None:
    """主函数"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 创建应用
        app = QApplication(sys.argv)
        
        # 初始化MediaPipe识别器
        logger.info("初始化MediaPipe识别器...")
        mp_recognizer = MediaPipeRecognizer()
        if not mp_recognizer.initialize():
            raise RuntimeError("MediaPipe识别器初始化失败")
            
        # 初始化自定义识别器
        logger.info("初始化自定义识别器...")
        custom_recognizer = CustomRecognizer()
        if not custom_recognizer.initialize():
            raise RuntimeError("自定义识别器初始化失败")
        
        # 创建主窗口，传入两个识别器（注意顺序：MediaPipe在前，自定义在后）
        window = RecognitionWindow(mp_recognizer, custom_recognizer)
        window.show()
        
        # 运行应用
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"运行出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 