#!/usr/bin/env python3
"""
可视化转换后的nnUNet数据集样本
用于验证转换是否正确
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_sample(dataset_path, case_idx=0):
    """
    可视化一个样本的图像和标签
    
    参数:
        dataset_path: 数据集路径
        case_idx: 样本索引
    """
    dataset_path = Path(dataset_path)
    
    # 查找对应的文件
    images_tr = list(dataset_path.glob('imagesTr/*_0000.png'))
    images_tr.sort()
    
    if case_idx >= len(images_tr):
        print(f"错误: 索引 {case_idx} 超出范围 (最大: {len(images_tr)-1})")
        return
    
    # 获取图像文件名
    image_file = images_tr[case_idx]
    # 从文件名中提取case名称（去掉最后的_0000后缀）
    case_name = '_'.join(image_file.stem.split('_')[:-1])
    label_file = dataset_path / 'labelsTr' / f'{case_name}.png'
    
    print(f"图像文件: {image_file}")
    print(f"标签文件: {label_file}")
    
    # 读取图像和标签
    img = Image.open(image_file)
    label = Image.open(label_file)
    
    img_array = np.array(img)
    label_array = np.array(label)
    
    print(f"图像尺寸: {img_array.shape}")
    print(f"标签尺寸: {label_array.shape}")
    print(f"标签中的唯一值: {np.unique(label_array)}")
    
    # 统计每个类别的像素数
    bg_count = np.sum(label_array == 0)
    pos_count = np.sum(label_array == 1)
    neg_count = np.sum(label_array == 2)
    
    print(f"\n类别统计:")
    print(f"  背景 (0): {bg_count} 像素")
    print(f"  阳性细胞 (1): {pos_count} 像素")
    print(f"  阴性细胞 (2): {neg_count} 像素")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 标签掩码
    axes[1].imshow(label_array, cmap='jet', vmin=0, vmax=2)
    axes[1].set_title('Label Mask\n(0=Background, 1=Positive, 2=Negative)')
    axes[1].axis('off')
    
    # 叠加显示
    axes[2].imshow(img_array)
    # 创建彩色掩码：阳性=红色，阴性=蓝色
    overlay = np.zeros((*label_array.shape, 4))
    overlay[label_array == 1] = [1, 0, 0, 0.5]  # 红色半透明 - 阳性
    overlay[label_array == 2] = [0, 0, 1, 0.5]  # 蓝色半透明 - 阴性
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay Display\n(Red=Positive, Blue=Negative)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    output_file = f'visualization_sample_{case_idx}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: {output_file}")
    
    # 显示（如果在GUI环境中）
    try:
        plt.show()
    except:
        pass


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化转换后的nnUNet数据集')
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='./nnUNet_raw/Dataset100_BCCellSegmentation',
        help='数据集路径'
    )
    parser.add_argument(
        '--index', 
        type=int, 
        default=0,
        help='要可视化的样本索引'
    )
    
    args = parser.parse_args()
    
    visualize_sample(args.dataset, args.index)


if __name__ == "__main__":
    main()

