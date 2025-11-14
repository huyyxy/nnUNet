#!/usr/bin/env python3
"""
可视化 SHIDC-B-Ki-67 数据集样本
用于查看标注的细胞核位置和类别
"""

import json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_sample(dataset_path, case_name=None, case_idx=0, subset='Train'):
    """
    可视化一个样本的图像和标注
    
    参数:
        dataset_path: 数据集路径 (SHIDC-B-Ki-67/bare images)
        case_name: 样本名称 (如 'p10_0031_5')，如果指定则忽略 case_idx
        case_idx: 样本索引 (当 case_name 为 None 时使用)
        subset: 子集名称 ('Train' 或 'Test')
    """
    dataset_path = Path(dataset_path)
    subset_path = dataset_path / subset
    
    if not subset_path.exists():
        print(f"错误: 路径 {subset_path} 不存在")
        return
    
    # 查找所有JSON文件
    json_files = list(subset_path.glob('*.json'))
    json_files.sort()
    
    if len(json_files) == 0:
        print(f"错误: 在 {subset_path} 中未找到 JSON 文件")
        return
    
    # 确定要可视化的文件
    if case_name:
        json_file = subset_path / f'{case_name}.json'
        image_file = subset_path / f'{case_name}.jpg'
        if not json_file.exists():
            print(f"错误: 文件 {json_file} 不存在")
            return
    else:
        if case_idx >= len(json_files):
            print(f"错误: 索引 {case_idx} 超出范围 (最大: {len(json_files)-1})")
            return
        json_file = json_files[case_idx]
        case_name = json_file.stem
        image_file = subset_path / f'{case_name}.jpg'
    
    print(f"样本名称: {case_name}")
    print(f"图像文件: {image_file}")
    print(f"标注文件: {json_file}")
    
    # 读取图像
    if not image_file.exists():
        print(f"错误: 图像文件 {image_file} 不存在")
        return
    
    img = Image.open(image_file)
    img_array = np.array(img)
    
    print(f"图像尺寸: {img_array.shape}")
    
    # 读取标注
    with open(json_file, 'r') as f:
        annotations = json.load(f)
    
    print(f"标注数量: {len(annotations)}")
    
    # 统计每个类别的数量
    label_counts = {1: 0, 2: 0, 3: 0}
    for ann in annotations:
        label_id = ann['label_id']
        if label_id in label_counts:
            label_counts[label_id] += 1
    
    print(f"\n类别统计:")
    print(f"  阳性染色细胞 (1): {label_counts[1]} 个")
    print(f"  阴性染色细胞 (2): {label_counts[2]} 个")
    print(f"  TIL细胞 (3): {label_counts[3]} 个")
    print(f"  总计: {sum(label_counts.values())} 个")
    
    # 定义颜色映射
    # 阳性=红色, 阴性=蓝色, TIL=绿色
    color_map = {
        1: (255, 0, 0),      # 红色 - 阳性染色
        2: (0, 0, 255),      # 蓝色 - 阴性染色
        3: (0, 255, 0)       # 绿色 - TIL
    }
    
    label_names = {
        1: 'Positive Stained',
        2: 'Negative Stained',
        3: 'TIL'
    }
    
    # 创建标注图像（在原图上绘制圆点）
    img_with_points = img.copy()
    draw = ImageDraw.Draw(img_with_points)
    
    # 绘制每个标注点
    point_radius = 8  # 点的半径
    for ann in annotations:
        x, y = ann['x'], ann['y']
        label_id = ann['label_id']
        color = color_map.get(label_id, (255, 255, 255))
        
        # 绘制圆点
        draw.ellipse(
            [(x - point_radius, y - point_radius), 
             (x + point_radius, y + point_radius)],
            outline=color,
            width=2
        )
    
    # 创建只有标注点的图像（黑色背景）
    points_only = Image.new('RGB', img.size, (0, 0, 0))
    draw_points = ImageDraw.Draw(points_only)
    
    for ann in annotations:
        x, y = ann['x'], ann['y']
        label_id = ann['label_id']
        color = color_map.get(label_id, (255, 255, 255))
        
        # 绘制实心圆点
        draw_points.ellipse(
            [(x - point_radius, y - point_radius), 
             (x + point_radius, y + point_radius)],
            fill=color,
            outline=color
        )
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图像
    axes[0].imshow(img_array)
    axes[0].set_title(f'Original Image\n{case_name}', fontsize=12)
    axes[0].axis('off')
    
    # 只有标注点
    axes[1].imshow(points_only)
    title_text = 'Annotated Points\n'
    title_text += f'Red={label_counts[1]} (Positive)\n'
    title_text += f'Blue={label_counts[2]} (Negative)\n'
    title_text += f'Green={label_counts[3]} (TIL)'
    axes[1].set_title(title_text, fontsize=11)
    axes[1].axis('off')
    
    # 叠加显示
    axes[2].imshow(img_with_points)
    axes[2].set_title('Overlay Display\n(Red=Positive, Blue=Negative, Green=TIL)', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    output_file = f'visualization_shidc_{case_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: {output_file}")
    
    # 显示（如果在GUI环境中）
    try:
        plt.show()
    except:
        pass


def list_samples(dataset_path, subset='Train', limit=10):
    """
    列出数据集中的样本
    
    参数:
        dataset_path: 数据集路径
        subset: 子集名称 ('Train' 或 'Test')
        limit: 显示的样本数量限制
    """
    dataset_path = Path(dataset_path)
    subset_path = dataset_path / subset
    
    if not subset_path.exists():
        print(f"错误: 路径 {subset_path} 不存在")
        return
    
    json_files = list(subset_path.glob('*.json'))
    json_files.sort()
    
    print(f"\n{subset} 集共有 {len(json_files)} 个样本")
    print(f"\n前 {min(limit, len(json_files))} 个样本:")
    
    for i, json_file in enumerate(json_files[:limit]):
        case_name = json_file.stem
        
        # 读取标注统计
        with open(json_file, 'r') as f:
            annotations = json.load(f)
        
        label_counts = {1: 0, 2: 0, 3: 0}
        for ann in annotations:
            label_id = ann['label_id']
            if label_id in label_counts:
                label_counts[label_id] += 1
        
        print(f"  [{i:3d}] {case_name:20s} - "
              f"阳性:{label_counts[1]:3d}, "
              f"阴性:{label_counts[2]:3d}, "
              f"TIL:{label_counts[3]:3d}, "
              f"总计:{sum(label_counts.values()):3d}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='可视化 SHIDC-B-Ki-67 数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 可视化 Train 集的第一个样本
  python visualize_shidc_ki67_data.py
  
  # 可视化指定索引的样本
  python visualize_shidc_ki67_data.py --index 5
  
  # 可视化指定名称的样本
  python visualize_shidc_ki67_data.py --name p10_0031_5
  
  # 可视化 Test 集的样本
  python visualize_shidc_ki67_data.py --subset Test --index 0
  
  # 列出所有样本
  python visualize_shidc_ki67_data.py --list --limit 20
        """
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='./SHIDC-B-Ki-67/bare images',
        help='数据集路径 (默认: ./SHIDC-B-Ki-67/bare images)'
    )
    
    parser.add_argument(
        '--subset',
        type=str,
        default='Train',
        choices=['Train', 'Test'],
        help='子集名称 (默认: Train)'
    )
    
    parser.add_argument(
        '--index', 
        type=int, 
        default=0,
        help='要可视化的样本索引 (默认: 0)'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='要可视化的样本名称 (如 p10_0031_5)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出数据集中的样本'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='列出样本时的数量限制 (默认: 10)'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_samples(args.dataset, args.subset, args.limit)
    else:
        visualize_sample(
            args.dataset, 
            case_name=args.name,
            case_idx=args.index,
            subset=args.subset
        )


if __name__ == "__main__":
    main()

