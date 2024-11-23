"""
@Author: blankxiao
@file: main.py
@Created: 2024-11-23
@Desc: 主程序入口
"""

import argparse
from typing import Optional, Any
import logging
import sys
from PyQt6.QtWidgets import QApplication
from ui.recognition_window import RecognitionWindow
from recognizers.custom_recognizer import CustomRecognizer
from recognizers.mediapipe_recognizer import MediaPipeRecognizer
from config import MODEL_CONFIG

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='手语识别系统')
    
    # 模型相关参数
    parser.add_argument('--model-path', type=str,
                      default=str(MODEL_CONFIG['model_save_dir'] / MODEL_CONFIG['model_name']),
                      help='自定义识别器使用的模型路径')
    parser.add_argument('--use-mediapipe', action='store_true',
                      help='是否使用MediaPipe识别器')
    parser.add_argument('--use-custom', action='store_true',
                      help='是否使用自定义识别器')
    
    args = parser.parse_args()
    
    # 如果都没指定，默认都使用
    if not args.use_mediapipe and not args.use_custom:
        args.use_mediapipe = True
        args.use_custom = True
    
    return args

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
    
    # 解析命令行参数
    args = parse_args()
    
    try:
        # 创建应用
        app = QApplication(sys.argv)
        
        recognizers = []
        
        # 根据参数初始化识别器
        if args.use_mediapipe:
            logger.info("初始化MediaPipe识别器...")
            mp_recognizer = MediaPipeRecognizer()
            if not mp_recognizer.initialize():
                raise RuntimeError("MediaPipe识别器初始化失败")
            recognizers.append(mp_recognizer)
            
        if args.use_custom:
            logger.info("初始化自定义识别器...")
            custom_recognizer = CustomRecognizer(model_path=args.model_path)
            if not custom_recognizer.initialize():
                raise RuntimeError("自定义识别器初始化失败")
            recognizers.append(custom_recognizer)
        
        if not recognizers:
            raise RuntimeError("没有可用的识别器")
        
        # 创建主窗口，传入识别器列表
        window = RecognitionWindow(*recognizers)
        window.show()
        
        # 运行应用
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"运行出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 