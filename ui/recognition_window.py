from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import cv2
import logging

logger = logging.getLogger(__name__)

class RecognitionWindow(QMainWindow):
    """识别窗口"""
    
    def __init__(self, mediapipe_recognizer, custom_recognizer=None):
        super().__init__()
        self.mediapipe_recognizer = mediapipe_recognizer
        self.custom_recognizer = custom_recognizer
        
        self.camera = None
        self.timer = None
        
        self.setup_ui()
        self.setup_camera()
        
    def setup_ui(self):
        """设置UI"""
        self.setWindowTitle("手势识别系统")
        self.setGeometry(100, 100, 1600, 800)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        central_widget.setLayout(main_layout)
        
        # MediaPipe识别器部分
        mediapipe_container = QWidget()
        mediapipe_layout = QVBoxLayout(mediapipe_container)
        mediapipe_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.mediapipe_video_label = QLabel()
        self.mediapipe_video_label.setMinimumSize(700, 600)
        self.mediapipe_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.mediapipe_result_label = QLabel("MediaPipe识别结果: 等待中...")
        self.mediapipe_result_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 5px;
                min-width: 200px;
            }
        """)
        self.mediapipe_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        mediapipe_layout.addWidget(self.mediapipe_video_label)
        mediapipe_layout.addWidget(self.mediapipe_result_label)
        
        main_layout.addWidget(mediapipe_container)
        
        # 自定义识别器部分（如果启用）
        if self.custom_recognizer:
            custom_container = QWidget()
            custom_layout = QVBoxLayout(custom_container)
            custom_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            self.custom_video_label = QLabel()
            self.custom_video_label.setMinimumSize(700, 600)
            self.custom_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            self.custom_result_label = QLabel("自定义识别结果: 等待中...")
            self.custom_result_label.setStyleSheet("""
                QLabel {
                    font-size: 18px;
                    padding: 10px;
                    background-color: #f0f0f0;
                    border-radius: 5px;
                    min-width: 200px;
                }
            """)
            self.custom_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            custom_layout.addWidget(self.custom_video_label)
            custom_layout.addWidget(self.custom_result_label)
            
            main_layout.addWidget(custom_container)
    
    def setup_camera(self):
        """设置摄像头和定时器"""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            logger.error("无法打开摄像头")
            return
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms 约等于 33fps
        
        # 启动识别器
        self.mediapipe_recognizer.start()
        if self.custom_recognizer:
            self.custom_recognizer.start()
    
    def update_frame(self):
        """更新视频帧"""
        ret, frame = self.camera.read()
        if not ret:
            return
        
        # 处理MediaPipe识别器
        mediapipe_frame, mediapipe_result = self.mediapipe_recognizer.process_frame(frame.copy())
        self.mediapipe_result_label.setText(f"MediaPipe识别结果: {mediapipe_result}")
        self.display_frame(mediapipe_frame, self.mediapipe_video_label)
        
        # 处理自定义识别器（如果启用）
        if self.custom_recognizer:
            custom_frame, custom_result = self.custom_recognizer.process_frame(frame.copy())
            self.custom_result_label.setText(f"自定义识别结果: {custom_result}")
            self.display_frame(custom_frame, self.custom_video_label)
    
    def display_frame(self, frame, label):
        """在标签上显示图像帧"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            label.width(), label.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        ))
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.camera:
            self.camera.release()
        if self.timer:
            self.timer.stop()
        super().closeEvent(event) 