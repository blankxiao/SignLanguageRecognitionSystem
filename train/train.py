"""
@Author: blankxiao
@file: train.py
@Created: 2024-11-22 23:10
@Desc: 训练模型
"""

import argparse
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import json

# 相对导入
from .dataset import SignLanguageDataset
from config import DATASET_CONFIG, MODEL_CONFIG

class SignLanguageResNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        
        # 简化网络结构，使用更小的通道数
        self.features = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 第二层卷积
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 第三层卷积
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Dropout用于防止过拟合
            nn.Dropout2d(0.25)
        )
        
        # 计算全连接层的输入维度
        # 64x64 经过3次MaxPool2d(2)后变成 8x8
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def calculate_metrics(y_true, y_pred, classes=range(10)):
    """计算各种评估指标"""
    # 计算精确率、召回率、F1分数
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=classes)
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    
    # 生成详细的分类报告
    class_report = classification_report(y_true, y_pred, labels=classes, output_dict=True)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }

def plot_confusion_matrix(conf_matrix, classes, save_path):
    """绘制混淆矩阵热力图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    fold: Optional[int] = None,
    writer: Optional[SummaryWriter] = None,
    epoch: Optional[int] = None
) -> Tuple[float, Dict]:
    """增强的验证函数，返回准确率和详细指标"""
    model.eval()
    val_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算所有指标
    metrics = calculate_metrics(all_labels, all_predictions)
    accuracy = 100 * sum(1 for x, y in zip(all_predictions, all_labels) if x == y) / len(all_labels)
    
    # 如果是在训练过程中，记录到TensorBoard
    if writer is not None and epoch is not None and fold is not None:
        # 记录每个类别的精确率、召回率和F1分数
        for i in range(10):  # 假设有10个类别
            writer.add_scalar(f'Fold_{fold+1}/Class_{i}/Precision', metrics['precision'][i], epoch)
            writer.add_scalar(f'Fold_{fold+1}/Class_{i}/Recall', metrics['recall'][i], epoch)
            writer.add_scalar(f'Fold_{fold+1}/Class_{i}/F1', metrics['f1'][i], epoch)
        
        # 记录整体指标
        writer.add_scalar(f'Fold_{fold+1}/Validation_Loss', val_loss / len(val_loader), epoch)
        writer.add_scalar(f'Fold_{fold+1}/Validation_Accuracy', accuracy, epoch)
    
    return accuracy, metrics

def setup_experiment_directory():
    """创建实验目录结构"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path('models') / timestamp
    
    # 创建各个子目录
    dirs = {
        'root': exp_dir,
        'checkpoints': exp_dir / 'checkpoints',  # 存放模型文件
        'logs': exp_dir / 'logs',  # TensorBoard日志
        'plots': exp_dir / 'plots',  # 存放混淆矩阵等图表
        'metrics': exp_dir / 'metrics'  # 存放评估指标
    }
    
    # 创建所有目录
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return dirs

def train_with_kfold(dataset, model_class, k_folds, args, device, writer, exp_dirs):
    """修改后的K折交叉验证训练函数"""
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=args.seed)
    all_fold_metrics = []
    indices = np.arange(len(dataset))
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(indices)):
        print(f'\n{"="*20} Fold {fold+1}/{k_folds} {"="*20}')
        
        # 创建数据加载器
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(dataset, batch_size=args.batch_size, 
                                sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, 
                              sampler=val_subsampler)
        
        # 为每个折叠创建新的模型实例
        model = model_class().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, 
                              weight_decay=MODEL_CONFIG['weight_decay'])
        
        # 更新模型保存路径
        model_save_path = exp_dirs['checkpoints'] / f'model_fold_{fold+1}.pth'
        
        # 训练当前折
        best_val_acc, best_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            writer=writer,
            num_epochs=args.epochs,
            device=device,
            fold=fold,
            model_save_path=model_save_path
        )
        
        # 保存混淆矩阵到plots目录
        plot_confusion_matrix(
            best_metrics['confusion_matrix'],
            classes=range(10),
            save_path=exp_dirs['plots'] / f'confusion_matrix_fold_{fold+1}.png'
        )
        
        # 保存当前折的详细指标
        fold_metrics_path = exp_dirs['metrics'] / f'metrics_fold_{fold+1}.json'
        with open(fold_metrics_path, 'w') as f:
            json.dump(best_metrics, f, indent=4)
        
        all_fold_metrics.append(best_metrics)
    
    # 保存总体评估报告
    with open(exp_dirs['metrics'] / 'detailed_evaluation.txt', 'w') as f:
        f.write(f'Detailed Evaluation Results (k={k_folds}):\n\n')
        
        # 记录每个折叠的详细指标
        for fold, metrics in enumerate(all_fold_metrics):
            f.write(f'\nFold {fold+1} Results:\n')
            f.write('-' * 50 + '\n')
            
            # 写入每个类别的指标
            f.write('\nPer-class metrics:\n')
            for i in range(10):
                f.write(f'\nClass {i}:\n')
                f.write(f'Precision: {metrics["precision"][i]:.4f}\n')
                f.write(f'Recall: {metrics["recall"][i]:.4f}\n')
                f.write(f'F1-score: {metrics["f1"][i]:.4f}\n')
                f.write(f'Support: {metrics["support"][i]}\n')
            
            # 写入分类报告
            f.write('\nClassification Report:\n')
            f.write(f'{classification_report(metrics["classification_report"])}\n')
        
        # 计算并记录平均指标
        avg_precision = np.mean([m['precision'] for m in all_fold_metrics], axis=0)
        avg_recall = np.mean([m['recall'] for m in all_fold_metrics], axis=0)
        avg_f1 = np.mean([m['f1'] for m in all_fold_metrics], axis=0)
        
        f.write('\nOverall Average Metrics:\n')
        f.write('-' * 50 + '\n')
        for i in range(10):
            f.write(f'\nClass {i} Averages:\n')
            f.write(f'Precision: {avg_precision[i]:.4f}\n')
            f.write(f'Recall: {avg_recall[i]:.4f}\n')
            f.write(f'F1-score: {avg_f1[i]:.4f}\n')

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    writer: SummaryWriter,
    num_epochs: int = 50,
    device: str = 'cuda',
    model_save_path: Optional[Path] = None
) -> float:
    """简化后的训练函数，只返回最佳验证准确率"""
    model = model.to(device)
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 10 == 9:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/10:.4f}')
                writer.add_scalar('Training Loss (per 10 batches)', 
                                running_loss/10, 
                                epoch * len(train_loader) + i)
                running_loss = 0.0
        
        train_acc = 100 * correct / total
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        # 记录指标
        writer.add_scalar('Training_Accuracy', train_acc, epoch)
        writer.add_scalar('Validation_Accuracy', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Accuracy: {val_acc:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}')
        print('-' * 50)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if model_save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        
        # 更新学习率
        scheduler.step(val_acc)
    
    return best_val_acc

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='手语识别模型训练脚本')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=MODEL_CONFIG['num_epochs'],
                      help='训练轮数 (默认: 100)')
    parser.add_argument('--batch-size', type=int, default=DATASET_CONFIG['batch_size'],
                      help='批次大小 (默认: 16)')
    parser.add_argument('--lr', type=float, default=MODEL_CONFIG['learning_rate'],
                      help='学习率 (默认: 0.0005)')
    
    # 模型相关参数
    parser.add_argument('--model-name', type=str, default=MODEL_CONFIG['model_name'],
                      help='模型保存名称 (默认: best_model.pth)')
    parser.add_argument('--model-dir', type=str, 
                      default=str(MODEL_CONFIG['model_save_dir']),
                      help='模型保存目录')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'],
                      help='训练设备 (默认: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子 (默认: 42)')
    
    # 添加交叉验证相关参数
    parser.add_argument('--k-folds', type=int, default=5,
                      help='交叉验证折数 (默认: 5)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置随机种子和设备
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 创建TensorBoard writer
    writer = SummaryWriter()
    
    # 加载数据集
    dataset = SignLanguageDataset(data_dir=str(DATASET_PATH))
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # 创建模型
    model = SignLanguageResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, 
                          weight_decay=MODEL_CONFIG['weight_decay'])
    
    # 训练模型
    best_val_acc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        writer=writer,
        num_epochs=args.epochs,
        device=device,
        model_save_path=Path(args.model_dir) / args.model_name
    )
    
    print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')
    writer.close()

if __name__ == "__main__":
    main() 