import sys
import logging
import argparse
from PyQt6.QtWidgets import QApplication
from ui.recognition_window import RecognitionWindow
from recognizers.mediapipe_recognizer import MediaPipeRecognizer
from recognizers.custom_recognizer import CustomRecognizer

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GestureRecognitionApp:
    """手势识别应用程序类"""
    
    def __init__(self, mode='dual'):
        self.app = QApplication(sys.argv)
        self.window = None
        self.mediapipe_recognizer = None
        self.custom_recognizer = None
        self.mode = mode
        
    def initialize(self):
        """初始化应用程序"""
        try:
            logger.info("初始化识别器...")
            self.mediapipe_recognizer = MediaPipeRecognizer()
            self.mediapipe_recognizer.initialize()
            
            if self.mode == 'dual':
                self.custom_recognizer = CustomRecognizer()
                self.custom_recognizer.initialize()
            
            logger.info("初始化窗口...")
            self.window = RecognitionWindow(
                mediapipe_recognizer=self.mediapipe_recognizer,
                custom_recognizer=self.custom_recognizer if self.mode == 'dual' else None
            )
            
        except Exception as e:
            logger.error(f"初始化失败: {str(e)}")
            raise
    
    def run(self):
        """运行应用程序"""
        try:
            self.initialize()
            self.window.show()
            return self.app.exec()
        except Exception as e:
            logger.error(f"运行出错: {str(e)}")
            return 1
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        if self.mediapipe_recognizer:
            self.mediapipe_recognizer.release()
        if self.custom_recognizer:
            self.custom_recognizer.release()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='手势识别系统')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'dual'],
        default='dual',
        help='运行模式：single(仅MediaPipe) 或 dual(MediaPipe和自定义)'
    )
    return parser.parse_args()

def main():
    """程序入口函数"""
    args = parse_args()
    app = GestureRecognitionApp(mode=args.mode)
    sys.exit(app.run())

if __name__ == "__main__":
    main() 