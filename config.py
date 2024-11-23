"""
@Author: blankxiao
@file: config.py
@Created: 2024-11-23
@Desc: 项目配置文件
"""

from pathlib import Path
from typing import Dict, Any

# 项目根目录
ROOT_DIR = Path(__file__).parent

# 数据集配置
DATASET_CONFIG = {
    'data_dir': ROOT_DIR / 'dataset' / 'sign-language-digits-dataset',
    'train_ratio': 0.8,
    'batch_size': 16
}

# 模型配置
MODEL_CONFIG = {
    'num_classes': 10,
    'learning_rate': 0.0005,
    'weight_decay': 0.001,
    'num_epochs': 100,
    'model_save_dir': ROOT_DIR / 'train' / 'models',
    'model_name': 'best_model.pth'
}

# 手部检测配置
HAND_DETECTION_CONFIG = {
    'min_hand_area': 5000,
    'max_hand_area': 50000,
    'padding': 20,
    'min_YCrCb': [0, 133, 77],
    'max_YCrCb': [255, 173, 127]
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
} 