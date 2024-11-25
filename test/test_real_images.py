"""
@Author: blankxiao
@file: test_real_images.py
@Created: 2024-11-23
@Desc: 测试真实图片的手势识别效果
"""

import torch
from pathlib import Path
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from train.train import SignLanguageResNet
from config import MODEL_CONFIG, TEST_CONFIG

def preprocess_image(image_path):
    """增强的图像预处理"""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # 自适应阈值处理
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 降噪
    img = cv2.medianBlur(img, 3)
    
    # 调整对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    # 调整大小
    img = cv2.resize(img, (64, 64))
    
    # 转换为tensor
    img = torch.FloatTensor(img).unsqueeze(0) / 255.0
    
    return img

def load_model(model_path):
    """加载训练好的模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignLanguageResNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

def predict_image(model, image_tensor, device):
    """预测单张图片"""
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # 添加batch维度
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
    return predicted.item(), confidence.item()

def visualize_prediction(image_path, prediction, confidence, save_dir):
    """可视化预测结果"""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f'Predicted: {prediction}\nConfidence: {confidence:.2f}')
    plt.axis('off')
    
    # 保存结果
    save_path = Path(save_dir) / f"{image_path.stem}_pred{prediction}.png"
    plt.savefig(save_path)
    plt.close()

def main():
    # 加载模型
    model_path = MODEL_CONFIG['model_save_dir'] / MODEL_CONFIG['model_name']
    model, device = load_model(model_path)
    print(f"Loaded model from {model_path}")
    
    # 设置测试图片目录
    test_img_dir = Path("test/testimg")
    if not test_img_dir.exists():
        raise ValueError(f"Test image directory not found: {test_img_dir}")
    
    # 创建结果保存目录
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    
    # 处理每张测试图片
    results = []
    for img_path in test_img_dir.glob("*.jpg"):  # 根据实际图片格式调整
        try:
            # 预处理图片
            img_tensor = preprocess_image(img_path)
            
            # 预测
            prediction, confidence = predict_image(model, img_tensor, device)
            
            # 记录结果
            result = {
                'image': img_path.name,
                'prediction': prediction,
                'confidence': confidence
            }
            results.append(result)
            
            # 可视化结果
            visualize_prediction(img_path, prediction, confidence, results_dir)
            
            print(f"Processed {img_path.name}:")
            print(f"Predicted: {prediction}")
            print(f"Confidence: {confidence:.2f}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # 保存所有结果到文本文件
    with open(results_dir / "predictions.txt", "w") as f:
        f.write("Test Results:\n\n")
        for result in results:
            f.write(f"Image: {result['image']}\n")
            f.write(f"Prediction: {result['prediction']}\n")
            f.write(f"Confidence: {result['confidence']:.2f}\n")
            f.write("-" * 50 + "\n")
    
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    main() 