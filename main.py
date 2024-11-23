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
        
        # 初始化识别器
        logger.info("初始化识别器...")
        recognizer = CustomRecognizer()
        if not recognizer.initialize():
            raise RuntimeError("识别器初始化失败")
        
        # 创建主窗口
        window = RecognitionWindow(recognizer)
        window.show()
        
        # 运行应用
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"运行出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 