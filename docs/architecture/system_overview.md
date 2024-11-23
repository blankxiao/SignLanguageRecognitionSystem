# 手势识别系统架构概述

## 1. 系统组成

系统由以下主要模块组成：

- UI界面模块：提供用户交互界面
- 手部检测模块：负责定位视频流中的手部区域
- 手势识别模块：对检测到的手部进行手势分类
- 训练模块：用于模型的训练和优化

## 2. 核心功能流程

1. **视频输入处理**
   - 从摄像头获取视频流
   - 对每一帧进行预处理

2. **手部检测**
   - 使用MediaPipe或自定义CV方法检测手部
   - 提取手部ROI区域
   - 应用预处理增强手部特征

3. **手势识别**
   - 使用训练好的深度学习模型进行分类
   - 输出预测结果和置信度
   - 支持实时识别和单帧识别

4. **结果展示**
   - 在UI界面实时显示检测框
   - 显示识别结果和置信度
   - 提供结果保存功能

## 3. 技术栈

- **UI**: PyQt6
- **视觉处理**: OpenCV
- **深度学习**: PyTorch
- **手部检测**: MediaPipe/OpenCV
- **数据处理**: NumPy/Pandas

## 4. 系统特点

- 支持多种手部检测方法
- 实时处理能力
- 模块化设计，易于扩展
- 提供详细的评估指标