"""
@Author: blankxiao
@file: train.py
@Created: 2024-11-22 23:10
@Desc: 训练模型
"""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from pathlib import Path

from .dataset import SignLanguageDataset

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

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    writer: SummaryWriter,
    num_epochs: int = 50,
    device: str = 'cuda'
) -> None:
    model = model.to(device)
    best_val_acc = 0.0
    patience = 15  # 增加耐心值
    patience_counter = 0
    
    # 使用更温和的学习率调度器
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
            
            # 减小L2正则化强度
            l2_lambda = 0.0001
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            
            loss.backward()
            
            # 更温和的梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
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
        val_acc = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_acc)
        
        # 记录指标
        writer.add_scalar('Training Accuracy', train_acc, epoch)
        writer.add_scalar('Validation Accuracy', val_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Accuracy: {val_acc:.2f}%')
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
            }, 'models/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    """单独的验证函数"""
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    return 100 * val_correct / val_total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建TensorBoard writer
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time)
    writer = SummaryWriter(log_dir)
    
    # 加载数据集
    dataset = SignLanguageDataset(data_dir="../dataset/sign-language-digits-dataset")
    
    # 划分数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器，减小batch size
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 初始化模型和训练组件
    model = SignLanguageResNet()
    criterion = nn.CrossEntropyLoss()
    # 使用更小的学习率
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001)
    
    # 确保models目录存在
    model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 修改模型保存路径
    model_save_path = os.path.join(model_save_dir, 'best_model.pth')
    
    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, writer, num_epochs=100, device=device)
    
    writer.close()

if __name__ == "__main__":
    main() 