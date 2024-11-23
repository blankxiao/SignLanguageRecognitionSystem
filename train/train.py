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
            
            nn.Dropout2d(0.25)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SEBlock(nn.Module):
    """Squeeze-and-Excitation块"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

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
                      help='模型保存名称 (默认: best_model.pth，会自动添加.pth后缀)')
    parser.add_argument('--model-dir', type=str, 
                      default=str(MODEL_CONFIG['model_save_dir']),
                      help='模型保存目录')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'],
                      help='训练设备 (默认: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子 (默认: 42)')
    
    return parser.parse_args()

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
    """训练模并返回最佳验证准确率"""
    model = model.to(device)
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    
    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # 初始周期
        T_mult=2  # 周期倍增
    )
    
    # 添加混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    def get_prediction_confidence(outputs):
        probs = torch.softmax(outputs, dim=1)
        confidence, _ = torch.max(probs, dim=1)
        return confidence.mean().item()
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 使用混合精度训练
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_confidences = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # 计算验证集的置信度
                val_confidences.append(get_prediction_confidence(outputs))
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # 更新学习率
        scheduler.step()
        
        # 记录指标
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 监控预测置信度
        train_confidence = get_prediction_confidence(outputs)
        val_confidence = np.mean(val_confidences)
        writer.add_scalar('Confidence/Train', train_confidence, epoch)
        writer.add_scalar('Confidence/Validation', val_confidence, epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}')
        print('-' * 50)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
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
    
    return best_val_acc

class LabelSmoothingLoss(nn.Module):
    """标签平滑损失函数"""
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

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
    dataset = SignLanguageDataset(data_dir=str(DATASET_CONFIG['data_dir']))
    
    # 划分训练集和验证集
    train_size = int(DATASET_CONFIG['train_ratio'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,  # 避免多进程导致问题
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 创建模型和优化器
    model = SignLanguageResNet().to(device)
    criterion = nn.CrossEntropyLoss()  # 使用普通的交叉熵损失
    optimizer = optim.Adam(  # 使用Adam优化器
        model.parameters(), 
        lr=args.lr,
        weight_decay=MODEL_CONFIG['weight_decay']
    )
    
    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # 确保模型保存目录存在
    model_save_dir = Path(args.model_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    # 确保模型文件名有.pth后缀
    model_name = args.model_name
    if not model_name.endswith('.pth'):
        model_name += '.pth'
    model_save_path = model_save_dir / model_name
    
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
        model_save_path=model_save_path
    )
    
    print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')
    writer.close()

if __name__ == "__main__":
    main() 