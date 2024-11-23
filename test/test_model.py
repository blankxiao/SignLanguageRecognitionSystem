"""
@Author: blankxiao
@file: test_model.py
@Created: 2024-11-23 01:00
@Desc: 测试训练好的手势识别模型
"""

from typing import Tuple, Optional
import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path

from train.train import SignLanguageResNet

def load_model(model_path: str) -> Tuple[SignLanguageResNet, torch.device]:
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignLanguageResNet()
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print("检查点内容：", checkpoint.keys())
        print("模型状态字典键：", checkpoint['model_state_dict'].keys())
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 打印一些模型信息
        print(f"设备: {device}")
        print(f"模型结构:\n{model}")
        
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        raise Exception(f"加载模型时出错: {str(e)}")

def preprocess_image(image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
    """预处理图像"""
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
        
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 添加自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 添加高斯模糊减少噪声
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    
    # 调整大小到64x64
    resized = cv2.resize(gray, (64, 64))
    
    # 增强对比度
    resized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
    
    # 归一化到[-1,1]范围，而不是[0,1]
    normalized = (resized.astype(np.float32) - 127.5) / 127.5
    
    # 转换为tensor
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor, img

def predict(
    model: SignLanguageResNet,
    image_tensor: torch.Tensor,
    device: torch.device
) -> Tuple[int, float]:
    """使用模型进行预测"""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
        confidence = probabilities.max().item()
        
        # 打印所有类别的概率
        print("\n各类别的概率分布：")
        for i, prob in enumerate(probabilities[0]):
            print(f"类别 {i}: {prob.item():.4f}")
            
    return prediction, confidence

def main():
    # 使用绝对路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'train', 'models', 'best_model.pth')
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testimg')
    
    # 打印路径信息以便调试
    print(f"当前工作目录: {os.getcwd()}")
    print(f"模型路径: {model_path}")
    print(f"测试图像目录: {test_dir}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在: {model_path}")
        print("请确保已经训练并保存了模型")
        return
        
    try:
        # 加载模型
        print("加载模型...")
        model, device = load_model(model_path)
        
        # 获取测试图像列表
        test_images = list(Path(test_dir).glob('*.jpg')) + list(Path(test_dir).glob('*.png'))
        
        if not test_images:
            print(f"在 {test_dir} 目录下没有找到测试图像")
            return
            
        # 处理每张测试图像
        for img_path in test_images:
            print(f"\n处理图像: {img_path.name}")
            
            # 预处理图像
            image_tensor, original_img = preprocess_image(str(img_path))
            
            # 进行预测
            prediction, confidence = predict(model, image_tensor, device)
            
            # 显示结果
            print(f"预测结果: {prediction}")
            print(f"置信度: {confidence:.2f}")
            
            # 在图像上显示结果
            result_img = original_img.copy()
            cv2.putText(result_img, f"Pred: {prediction} ({confidence:.2f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示图像
            cv2.imshow('Result', result_img)
            key = cv2.waitKey(0)
            
            # 按'q'退出
            if key == ord('q'):
                break
                
    except Exception as e:
        print(f"测试过程中出错: {str(e)}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 