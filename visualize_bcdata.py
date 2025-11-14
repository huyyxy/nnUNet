#!/usr/bin/env python3
"""
可视化 BCData 数据集样本
用于查看细胞图像及其阳性/阴性细胞标注
"""

import h5py
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pathlib import Path


def load_coordinates_from_h5(h5_file):
    """
    从.h5文件加载坐标
    
    参数:
        h5_file: .h5文件路径
    
    返回:
        coords: numpy数组，形状为(N, 2)，格式为(x, y)
    """
    if not h5_file.exists():
        return np.array([]).reshape(0, 2)
    
    try:
        with h5py.File(h5_file, 'r') as f:
            if 'coordinates' in f:
                coords = f['coordinates'][...]
                return coords
            else:
                return np.array([]).reshape(0, 2)
    except Exception as e:
        print(f"警告: 读取文件 {h5_file} 时出错: {e}")
        return np.array([]).reshape(0, 2)


def visualize_sample(dataset_path, case_name=None, case_idx=0, subset='train'):
    """
    可视化一个样本的图像和标注
    
    参数:
        dataset_path: 数据集路径 (BCData)
        case_name: 样本名称 (如 '0', '100')，如果指定则忽略 case_idx
        case_idx: 样本索引 (当 case_name 为 None 时使用)
        subset: 子集名称 ('train', 'validation', 或 'test')
    """
    dataset_path = Path(dataset_path)
    images_path = dataset_path / 'images' / subset
    annotations_path = dataset_path / 'annotations' / subset
    
    if not images_path.exists():
        print(f"错误: 路径 {images_path} 不存在")
        return
    
    # 查找所有PNG文件
    image_files = sorted(list(images_path.glob('*.png')), key=lambda x: int(x.stem))
    
    if len(image_files) == 0:
        print(f"错误: 在 {images_path} 中未找到 PNG 文件")
        return
    
    # 确定要可视化的文件
    if case_name:
        image_file = images_path / f'{case_name}.png'
        if not image_file.exists():
            print(f"错误: 文件 {image_file} 不存在")
            return
    else:
        if case_idx >= len(image_files):
            print(f"错误: 索引 {case_idx} 超出范围 (最大: {len(image_files)-1})")
            return
        image_file = image_files[case_idx]
        case_name = image_file.stem
    
    print(f"样本名称: {case_name}")
    print(f"图像文件: {image_file}")
    print(f"子集: {subset}")
    
    # 读取图像
    img = Image.open(image_file)
    img_array = np.array(img)
    
    print(f"图像尺寸: {img_array.shape}")
    
    # 读取阳性和阴性细胞坐标
    positive_h5 = annotations_path / 'positive' / f'{case_name}.h5'
    negative_h5 = annotations_path / 'negative' / f'{case_name}.h5'
    
    print(f"阳性标注: {positive_h5}")
    print(f"阴性标注: {negative_h5}")
    
    positive_coords = load_coordinates_from_h5(positive_h5)
    negative_coords = load_coordinates_from_h5(negative_h5)
    
    num_positive = len(positive_coords)
    num_negative = len(negative_coords)
    num_total = num_positive + num_negative
    
    print(f"\n类别统计:")
    print(f"  阳性细胞 (Positive): {num_positive} 个")
    print(f"  阴性细胞 (Negative): {num_negative} 个")
    print(f"  总计: {num_total} 个")
    
    if num_total > 0:
        ki67_index = (num_positive / num_total) * 100
        print(f"  Ki-67增殖指数: {ki67_index:.2f}%")
    
    # 定义颜色
    # 阳性=红色, 阴性=蓝色
    positive_color = (255, 0, 0)  # 红色
    negative_color = (0, 0, 255)  # 蓝色
    
    # 创建标注图像（在原图上绘制圆点）
    img_with_points = img.copy()
    draw = ImageDraw.Draw(img_with_points)
    
    point_radius = 6  # 点的半径
    
    # 绘制阴性细胞（蓝色）
    for coord in negative_coords:
        x, y = int(coord[0]), int(coord[1])
        draw.ellipse(
            [(x - point_radius, y - point_radius), 
             (x + point_radius, y + point_radius)],
            outline=negative_color,
            width=2
        )
    
    # 绘制阳性细胞（红色）
    for coord in positive_coords:
        x, y = int(coord[0]), int(coord[1])
        draw.ellipse(
            [(x - point_radius, y - point_radius), 
             (x + point_radius, y + point_radius)],
            outline=positive_color,
            width=2
        )
    
    # 创建只有标注点的图像（黑色背景）
    points_only = Image.new('RGB', img.size, (0, 0, 0))
    draw_points = ImageDraw.Draw(points_only)
    
    # 绘制阴性细胞（蓝色实心圆）
    for coord in negative_coords:
        x, y = int(coord[0]), int(coord[1])
        draw_points.ellipse(
            [(x - point_radius, y - point_radius), 
             (x + point_radius, y + point_radius)],
            fill=negative_color,
            outline=negative_color
        )
    
    # 绘制阳性细胞（红色实心圆）
    for coord in positive_coords:
        x, y = int(coord[0]), int(coord[1])
        draw_points.ellipse(
            [(x - point_radius, y - point_radius), 
             (x + point_radius, y + point_radius)],
            fill=positive_color,
            outline=positive_color
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
    title_text += f'Red={num_positive} (Positive)\n'
    title_text += f'Blue={num_negative} (Negative)\n'
    if num_total > 0:
        title_text += f'Ki-67 Index={ki67_index:.1f}%'
    axes[1].set_title(title_text, fontsize=11)
    axes[1].axis('off')
    
    # 叠加显示
    axes[2].imshow(img_with_points)
    axes[2].set_title('Overlay Display\n(Red=Positive, Blue=Negative)', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    output_file = f'visualization_bcdata_{subset}_{case_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到: {output_file}")
    
    # 显示（如果在GUI环境中）
    try:
        plt.show()
    except:
        pass
    
    plt.close()


def list_samples(dataset_path, subset='train', limit=10):
    """
    列出数据集中的样本
    
    参数:
        dataset_path: 数据集路径
        subset: 子集名称 ('train', 'validation', 或 'test')
        limit: 显示的样本数量限制
    """
    dataset_path = Path(dataset_path)
    images_path = dataset_path / 'images' / subset
    annotations_path = dataset_path / 'annotations' / subset
    
    if not images_path.exists():
        print(f"错误: 路径 {images_path} 不存在")
        return
    
    image_files = sorted(list(images_path.glob('*.png')), key=lambda x: int(x.stem))
    
    print(f"\n{subset} 集共有 {len(image_files)} 个样本")
    print(f"\n前 {min(limit, len(image_files))} 个样本:")
    
    for i, image_file in enumerate(image_files[:limit]):
        case_name = image_file.stem
        
        # 读取标注统计
        positive_h5 = annotations_path / 'positive' / f'{case_name}.h5'
        negative_h5 = annotations_path / 'negative' / f'{case_name}.h5'
        
        positive_coords = load_coordinates_from_h5(positive_h5)
        negative_coords = load_coordinates_from_h5(negative_h5)
        
        num_positive = len(positive_coords)
        num_negative = len(negative_coords)
        num_total = num_positive + num_negative
        
        ki67_index = (num_positive / num_total * 100) if num_total > 0 else 0.0
        
        print(f"  [{i:4d}] {case_name:10s} - "
              f"Positive:{num_positive:4d}, "
              f"Negative:{num_negative:4d}, "
              f"Total:{num_total:4d}, "
              f"Ki-67:{ki67_index:6.2f}%")


def statistics_summary(dataset_path, subset='train'):
    """
    统计数据集的整体信息
    
    参数:
        dataset_path: 数据集路径
        subset: 子集名称 ('train', 'validation', 或 'test')
    """
    dataset_path = Path(dataset_path)
    images_path = dataset_path / 'images' / subset
    annotations_path = dataset_path / 'annotations' / subset
    
    if not images_path.exists():
        print(f"错误: 路径 {images_path} 不存在")
        return
    
    image_files = sorted(list(images_path.glob('*.png')), key=lambda x: int(x.stem))
    
    print(f"\n{'='*60}")
    print(f"{subset.upper()} 集统计信息")
    print(f"{'='*60}")
    print(f"样本总数: {len(image_files)}")
    
    total_positive = 0
    total_negative = 0
    ki67_indices = []
    
    for image_file in image_files:
        case_name = image_file.stem
        
        positive_h5 = annotations_path / 'positive' / f'{case_name}.h5'
        negative_h5 = annotations_path / 'negative' / f'{case_name}.h5'
        
        positive_coords = load_coordinates_from_h5(positive_h5)
        negative_coords = load_coordinates_from_h5(negative_h5)
        
        num_positive = len(positive_coords)
        num_negative = len(negative_coords)
        num_total = num_positive + num_negative
        
        total_positive += num_positive
        total_negative += num_negative
        
        if num_total > 0:
            ki67_indices.append(num_positive / num_total * 100)
    
    total_cells = total_positive + total_negative
    
    print(f"\n细胞统计:")
    print(f"  阳性细胞总数: {total_positive}")
    print(f"  阴性细胞总数: {total_negative}")
    print(f"  细胞总数: {total_cells}")
    print(f"  平均每张图像的细胞数: {total_cells / len(image_files):.1f}")
    
    if ki67_indices:
        ki67_indices = np.array(ki67_indices)
        print(f"\nKi-67增殖指数统计:")
        print(f"  平均值: {np.mean(ki67_indices):.2f}%")
        print(f"  中位数: {np.median(ki67_indices):.2f}%")
        print(f"  标准差: {np.std(ki67_indices):.2f}%")
        print(f"  最小值: {np.min(ki67_indices):.2f}%")
        print(f"  最大值: {np.max(ki67_indices):.2f}%")
    
    print(f"{'='*60}\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='可视化 BCData 数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 可视化 train 集的第一个样本
  python visualize_bcdata.py
  
  # 可视化指定索引的样本
  python visualize_bcdata.py --index 5
  
  # 可视化指定名称的样本
  python visualize_bcdata.py --name 100
  
  # 可视化 validation 集的样本
  python visualize_bcdata.py --subset validation --index 0
  
  # 列出所有样本
  python visualize_bcdata.py --list --limit 20
  
  # 显示数据集统计信息
  python visualize_bcdata.py --stats
        """
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='./BCData',
        help='数据集路径 (默认: ./BCData)'
    )
    
    parser.add_argument(
        '--subset',
        type=str,
        default='train',
        choices=['train', 'validation', 'test'],
        help='子集名称 (默认: train)'
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
        help='要可视化的样本名称 (如 0, 100)'
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
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='显示数据集统计信息'
    )
    
    args = parser.parse_args()
    
    if args.stats:
        statistics_summary(args.dataset, args.subset)
    elif args.list:
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

