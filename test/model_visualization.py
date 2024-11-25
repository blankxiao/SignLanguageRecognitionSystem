"""
@Author: blankxiao
@file: model_visualization.py
@Created: 2024-11-23
@Desc: 模型处理过程可视化，展示特征提取的各个阶段
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from train.train import SignLanguageResNet
import matplotlib.pyplot as plt
from typing import Dict, List
import seaborn as sns
import argparse
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor(nn.Module):
    """特征提取器，用于获取中间层的输出"""
    def __init__(self, model: nn.Module, layers: List[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self.features = {}
        
        # 注册钩子，获取中间层输出
        for layer_name in layers:
            layer = dict([*self.model.named_modules()])[layer_name]
            layer.register_forward_hook(self.get_feature_hook(layer_name))
    
    def get_feature_hook(self, layer_name: str):
        def hook(module, input, output):
            self.features[layer_name] = output.detach()
        return hook
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.model(x)
        return self.features

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='模型特征可视化工具')
    
    parser.add_argument('--model-path', type=str, required=True,
                      help='模型文件路径 (.pth)')
    parser.add_argument('--image-path', type=str, required=True,
                      help='测试图片路径')
    parser.add_argument('--save-dir', type=str, default='test_results',
                      help='结果保存目录 (默认: test_results)')
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'],
                      help='运行设备 (默认: cuda)')
    
    return parser.parse_args()

def visualize_features(args):
    """可视化模型处理过程"""
    try:
        # 检查文件是否存在
        if not Path(args.model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
        if not Path(args.image_path).exists():
            raise FileNotFoundError(f"图片文件不存在: {args.image_path}")
        
        # 创建保存目录
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置设备
        device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
        logger.info(f"使用设备: {device}")
        
        # 加载模型
        model = SignLanguageResNet().to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info(f"模型加载成功，验证准确率: {checkpoint.get('val_acc', 'N/A')}")
        
        # 创建特征提取器
        layers_to_extract = [
            'features.0',   # 第一个卷积层
            'features.4',   # 第一个卷积块后
            'features.8',   # 第二个卷积块后
            'features.12',  # 第三个卷积块后
        ]
        extractor = FeatureExtractor(model, layers_to_extract)
        
        # 读取并预处理图像
        image = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
        original_image = image.copy()
        image = cv2.resize(image, (64, 64))
        image = (image.astype(np.float32) - 127.5) / 127.5
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
        
        # 获取特征图
        with torch.no_grad():
            features = extractor(tensor)
            outputs = model(tensor)
        
        # 创建可视化图像
        fig = plt.figure(figsize=(20, 10))
        
        # 1. 显示原始图像
        plt.subplot(2, 3, 1)
        plt.title('Input Image', fontsize=12)
        plt.imshow(original_image, cmap='gray')
        plt.axis('off')
        
        # 2. 显示各层特征图的热力图
        layer_names = {
            'features.0': 'First Conv Layer',
            'features.4': 'After First Block',
            'features.8': 'After Second Block',
            'features.12': 'Final Features'
        }
        
        for idx, (layer_name, feature) in enumerate(features.items()):
            # 选择该层的所有特征图
            feature_maps = feature[0].cpu().numpy()
            
            # 计算特征图的激活强度
            activation_map = np.mean(np.abs(feature_maps), axis=0)
            activation_map = cv2.resize(activation_map, (original_image.shape[1], original_image.shape[0]))
            
            # 归一化到[0,1]范围
            activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
            
            # 创建热力图
            plt.subplot(2, 3, idx + 2)
            plt.title(layer_names[layer_name], fontsize=12)
            plt.imshow(original_image, cmap='gray')
            heatmap = plt.imshow(activation_map, cmap='jet', alpha=0.5)
            plt.colorbar(heatmap, label='Activation Strength')
            plt.axis('off')
        
        # 3. 显示预测结果
        probabilities = torch.softmax(outputs, dim=1)[0]
        prediction = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[prediction].item()
        
        plt.subplot(2, 3, 6)
        plt.title('Prediction Probabilities', fontsize=12)
        sns.barplot(x=list(range(10)), y=probabilities.cpu().numpy())
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        
        # 添加总标题
        plt.suptitle(f'Feature Visualization\nPredicted: {prediction} (Confidence: {confidence:.2f})',
                    fontsize=14)
        
        # 调整布局并保存
        plt.tight_layout()
        save_path = save_dir / f'feature_visualization_{Path(args.image_path).stem}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"可视化结果已保存到: {save_path}")
        plt.show()
        
    except Exception as e:
        logger.error(f"可视化过程出错: {str(e)}")

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 解析命令行参数并运行
    args = parse_args()
    visualize_features(args) 