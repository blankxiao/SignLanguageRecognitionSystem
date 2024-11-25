"""
@Author: blankxiao
@file: show_model.py
@Created: 2024-11-26
@Desc: 模型结构可视化工具
"""

import torch
import torch.nn as nn
from train.train import SignLanguageResNet
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
from graphviz import Digraph
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

logger = logging.getLogger(__name__)

def export_model_for_netron(model: nn.Module, save_path: str, device='cpu'):
    """导出模型为ONNX格式，供Netron查看"""
    try:
        # 检查是否安装了onnx
        import onnx
        
        # 确保模型和输入在同一设备上
        model = model.to(device)
        dummy_input = torch.randn(1, 1, 64, 64, device=device)
        
        # 导出模型
        torch.onnx.export(
            model,                  # 模型
            dummy_input,           # 示例输入
            save_path,             # 输出文件名
            verbose=True,          # 显示详细信息
            input_names=['input'], # 输入节点名称
            output_names=['output']# 输出节点名称
        )
        logger.info(f"模型已导出到: {save_path}")
    except ImportError:
        logger.warning("ONNX模块未安装，跳过ONNX导出。可以使用 'pip install onnx' 安装")
    except Exception as e:
        logger.error(f"导出模型失败: {str(e)}")

def visualize_model_graph(model: nn.Module, save_path: str, device='cpu'):
    """使用torchviz可视化模型计算图"""
    try:
        # 确保模型和输入在同一设备上
        model = model.to(device)
        x = torch.randn(1, 1, 64, 64, device=device)
        y = model(x)
        dot = make_dot(y, params=dict(model.named_parameters()))
        dot.render(save_path, format='png')
        logger.info(f"计算图已保存到: {save_path}.png")
    except Exception as e:
        logger.error(f"生成计算图失败: {str(e)}")

def analyze_model_parameters(model: nn.Module):
    """分析模型参数统计信息"""
    total_params = 0
    trainable_params = 0
    param_sizes = {}
    
    for name, param in model.named_parameters():
        param_size = param.numel()
        total_params += param_size
        if param.requires_grad:
            trainable_params += param_size
        layer_name = name.split('.')[0]
        param_sizes[layer_name] = param_sizes.get(layer_name, 0) + param_size
    
    # 打印统计信息
    print("\n模型参数统计:")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"固定参数量: {total_params - trainable_params:,}")
    
    # 绘制参数分布图
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(param_sizes.keys()), y=list(param_sizes.values()))
    plt.title("各层参数数量分布")
    plt.xticks(rotation=45)
    plt.ylabel("参数数量")
    plt.tight_layout()
    plt.savefig("param_distribution.png")
    plt.close()

def visualize_with_tensorboard(model: nn.Module, log_dir: str, device='cpu'):
    """使用TensorBoard可视化模型结构"""
    writer = SummaryWriter(log_dir)
    dummy_input = torch.randn(1, 1, 64, 64, device=device)
    model = model.to(device)
    writer.add_graph(model, dummy_input)
    writer.close()
    logger.info(f"模型结构已写入TensorBoard，运行 tensorboard --logdir={log_dir} 查看")

def visualize_model_structure(model: nn.Module, save_path: str):
    """使用ONNX和Netron可视化模型结构"""
    try:
        # 导出ONNX模型
        dummy_input = torch.randn(1, 1, 64, 64)  # 批次大小=1, 通道=1, 高=64, 宽=64
        torch.onnx.export(
            model,               # 模型
            dummy_input,        # 示例输入
            save_path,          # 保存路径
            input_names=['image'],
            output_names=['prediction'],
            dynamic_axes={
                'image': {0: 'batch_size'},    # 批次大小可变
                'prediction': {0: 'batch_size'} # 输出也跟随批次大小变化
            }
        )
        logger.info(f"模型已导出为ONNX格式: {save_path}")
        logger.info("请使用 Netron (https://netron.app) 查看模型结构")
        
        # 可选：使用onnx验证模型
        import onnx
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX模型验证通过")
        
    except ImportError:
        logger.warning("请安装ONNX: pip install onnx")
    except Exception as e:
        logger.error(f"模型导出失败: {str(e)}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='模型结构可视化工具')
    parser.add_argument('--model-path', type=str, required=True,
                      help='模型文件路径 (.pth)')
    parser.add_argument('--save-dir', type=str, default='model_visualization',
                      help='可视化结果保存目录')
    parser.add_argument('--device', type=str, default='cpu',
                      choices=['cuda', 'cpu'],
                      help='运行设备')
    return parser.parse_args()

def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 解析参数
    args = parse_args()
    
    try:
        # 创建保存目录
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置设备
        device = args.device
        
        # 加载模型
        model = SignLanguageResNet()
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # 1. 导出ONNX模型
        export_model_for_netron(
            model, 
            str(save_dir / 'model.onnx'),
            device=device
        )
        
        # 2. 生成计算图
        visualize_model_graph(
            model,
            str(save_dir / 'model_graph'),
            device=device
        )
        
        # 3. 分析参数统计
        analyze_model_parameters(model)
        
        # 4. TensorBoard可视化
        visualize_with_tensorboard(
            model,
            str(save_dir / 'tensorboard_logs'),
            device=device
        )
        
        # 可视化模型结构
        visualize_model_structure(
            model,
            str(save_dir / 'model_structure')
        )
        
        logger.info("模型可视化完成！")
        
    except Exception as e:
        logger.error(f"可视化过程出错: {str(e)}")

if __name__ == "__main__":
    main()