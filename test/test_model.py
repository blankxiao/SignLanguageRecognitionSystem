"""
@Author: blankxiao
@file: test_model.py
@Created: 2024-11-23 01:00
@Desc: 测试训练好的手势识别模型
"""

import torch
from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import random

from train.train import SignLanguageResNet
from train.dataset import SignLanguageDataset
from config import DATASET_CONFIG, MODEL_CONFIG, TEST_CONFIG

def load_model(model_path):
    """加载训练好的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignLanguageResNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

def test_on_dataset(model, test_loader, device):
    """在测试集上评估模型"""
    all_preds = []
    all_labels = []
    all_confidences = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            # 计算置信度
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
    
    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels, all_confidences

def plot_confusion_matrix(y_true, y_pred, save_path):
    """绘制并保存混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def analyze_errors(model, test_loader, device, save_dir):
    """分析错误预测的案例"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    error_cases = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            
            # 找出预测错误的样本
            errors = predicted != labels
            for i in range(len(errors)):
                if errors[i]:
                    error_cases.append({
                        'image': images[i].cpu().numpy(),
                        'true_label': labels[i].item(),
                        'predicted': predicted[i].item(),
                        'confidence': confidence[i].item()
                    })
    
    # 保存错误案例分析报告
    with open(save_dir / 'error_analysis.txt', 'w') as f:
        f.write(f"Total error cases: {len(error_cases)}\n\n")
        for i, case in enumerate(error_cases):
            f.write(f"Case {i+1}:\n")
            f.write(f"True label: {case['true_label']}\n")
            f.write(f"Predicted: {case['predicted']}\n")
            f.write(f"Confidence: {case['confidence']:.4f}\n")
            f.write("-" * 50 + "\n")
    
    return error_cases

def main():
    # 加载模型
    model_path = MODEL_CONFIG['model_save_dir'] / MODEL_CONFIG['model_name']
    model, device = load_model(model_path)
    print(f"Loaded model from {model_path}")
    
    # 创建测试数据集
    dataset = SignLanguageDataset(data_dir=str(DATASET_CONFIG['data_dir']))
    
    # 随机选择20%的数据作为测试集
    test_size = int(0.2 * len(dataset))
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    test_indices = indices[:test_size]
    
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=TEST_CONFIG['test_batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # 在测试集上评估
    accuracy, all_preds, all_labels, all_confidences = test_on_dataset(
        model, test_loader, device
    )
    print(f"\nTest Results:")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Average Confidence: {np.mean(all_confidences):.4f}")
    
    # 打印详细的分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    
    # 确保error_cases目录存在
    error_cases_dir = Path(TEST_CONFIG['error_cases_dir'])
    error_cases_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        all_labels, 
        all_preds, 
        save_path=error_cases_dir / 'confusion_matrix.png'
    )
    
    # 分析错误案例
    if TEST_CONFIG['save_error_cases']:
        error_cases = analyze_errors(
            model,
            test_loader,
            device,
            error_cases_dir
        )
        print(f"\nFound {len(error_cases)} error cases.")
        print(f"Error analysis saved to {error_cases_dir}")

if __name__ == '__main__':
    main() 