"""
@Author: blankxiao
@file: test_model.py
@Created: 2024-11-23 01:00
@Desc: 测试训练好的手势识别模型
"""

import argparse
from typing import Tuple, Optional, Dict
import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from train.train import SignLanguageResNet
from config import MODEL_CONFIG

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='手语识别模型测试脚本')
    
    # 模型相关参数
    parser.add_argument('--model-path', type=str,
                      default=str(MODEL_CONFIG['model_save_dir'] / MODEL_CONFIG['model_name']),
                      help='模型文件路径')
    parser.add_argument('--test-dir', type=str,
                      default='test/testimg',
                      help='测试图像目录')
    
    # 测试相关参数
    parser.add_argument('--batch-size', type=int, default=1,
                      help='批处理大小')
    parser.add_argument('--device', type=str,
                      choices=['cuda', 'cpu'], default='cuda',
                      help='使用设备')
    parser.add_argument('--save-results', action='store_true',
                      help='是否保存测试结果')
    parser.add_argument('--output-dir', type=str,
                      default='test_results',
                      help='测试结果保存目录')
    parser.add_argument('--show-images', action='store_true',
                      help='是否显示测试图像')
    
    return parser.parse_args()

def save_test_results(
    predictions: list,
    confidences: list,
    image_paths: list,
    output_dir: str,
    metrics: Optional[Dict] = None
):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存预测结果
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write("测试结果报告\n")
        f.write("=" * 50 + "\n\n")
        
        for img_path, pred, conf in zip(image_paths, predictions, confidences):
            f.write(f"图像: {os.path.basename(img_path)}\n")
            f.write(f"预测类别: {pred}\n")
            f.write(f"置信度: {conf:.4f}\n")
            f.write("-" * 30 + "\n")
        
        # 如果有评估指标，也保存它们
        if metrics:
            f.write("\n性能评估指标:\n")
            f.write("=" * 50 + "\n")
            f.write(f"分类报告:\n{metrics['classification_report']}\n")
            
            # 保存每个类别的详细指标
            f.write("\n每个类别的详细指标:\n")
            for i in range(len(metrics['precision'])):
                f.write(f"\n类别 {i}:\n")
                f.write(f"精确率: {metrics['precision'][i]:.4f}\n")
                f.write(f"召回率: {metrics['recall'][i]:.4f}\n")
                f.write(f"F1分数: {metrics['f1'][i]:.4f}\n")
                f.write(f"支持度: {metrics['support'][i]}\n")

def plot_and_save_confusion_matrix(conf_matrix: np.ndarray, save_path: str):
    """绘制并保存混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_model(model_path: str, device: str) -> Tuple[SignLanguageResNet, torch.device]:
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = SignLanguageResNet()
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型加载自: {model_path}")
        print(f"验证准确率: {checkpoint.get('val_acc', 'N/A')}")
        if 'metrics' in checkpoint:
            print("模型评估指标:")
            print(f"F1分数: {np.mean(checkpoint['metrics']['f1']):.4f}")
        
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        raise Exception(f"加载模型时出错: {str(e)}")

def main():
    args = parse_args()
    
    try:
        # 加载模型
        print("加载模型...")
        model, device = load_model(args.model_path, args.device)
        
        # 获取测试图像列表
        test_dir = Path(args.test_dir)
        test_images = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
        
        if not test_images:
            print(f"在 {test_dir} 目录下没有找到测试图像")
            return
        
        # 存储预测结果
        predictions = []
        confidences = []
        true_labels = []  # 如果有真实标签的话
        
        # 处理每张测试图像
        for img_path in test_images:
            print(f"\n处理图像: {img_path.name}")
            
            # 预处理图像
            image_tensor, original_img = preprocess_image(str(img_path))
            
            # 进行预测
            prediction, confidence = predict(model, image_tensor, device)
            predictions.append(prediction)
            confidences.append(confidence)
            
            # 如果需要显示图像
            if args.show_images:
                result_img = original_img.copy()
                cv2.putText(result_img, f"Pred: {prediction} ({confidence:.2f})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Result', result_img)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
        
        # 如果需要保存结果
        if args.save_results:
            # 计算评估指标（如果有真实标签）
            metrics = None
            if true_labels:
                metrics = {
                    'precision': precision_recall_fscore_support(true_labels, predictions)[0],
                    'recall': precision_recall_fscore_support(true_labels, predictions)[1],
                    'f1': precision_recall_fscore_support(true_labels, predictions)[2],
                    'support': precision_recall_fscore_support(true_labels, predictions)[3],
                    'confusion_matrix': confusion_matrix(true_labels, predictions),
                    'classification_report': classification_report(true_labels, predictions)
                }
                
                # 保存混淆矩阵
                plot_and_save_confusion_matrix(
                    metrics['confusion_matrix'],
                    os.path.join(args.output_dir, 'confusion_matrix.png')
                )
            
            # 保存测试结果
            save_test_results(
                predictions=predictions,
                confidences=confidences,
                image_paths=[str(p) for p in test_images],
                output_dir=args.output_dir,
                metrics=metrics
            )
            
            print(f"\n测试结果已保存到: {args.output_dir}")
    
    except Exception as e:
        print(f"测试过程中出错: {str(e)}")
    finally:
        if args.show_images:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 