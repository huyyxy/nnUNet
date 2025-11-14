#!/usr/bin/env python3
"""
将BCData数据集转换为nnUNetv2格式
BCData包含：
- images/: RGB图像 (640x640)
- annotations/: .h5文件包含阳性和阴性细胞的(x,y)坐标

转换为nnUNetv2格式：
- imagesTr/: 训练图像,图像名称为{dataset_name}_{case_id:04d}_0000.png,其中case_id为训练样本的编号,640x640 RGB格式的png图像
- labelsTr/: 对应的分割标签（0=背景, 1=阳性细胞, 2=阴性细胞）
- dataset.json: 元数据
"""

import os
import h5py
import numpy as np
from PIL import Image, ImageDraw
import json
from pathlib import Path
from tqdm import tqdm
import shutil


def create_segmentation_mask(image_shape, positive_coords, negative_coords, cell_radius=3):
    """
    根据细胞坐标创建分割掩码
    
    参数:
        image_shape: (height, width) 图像尺寸
        positive_coords: 阳性细胞坐标 array of shape (N, 2) 格式为 (x, y)
        negative_coords: 阴性细胞坐标 array of shape (M, 2) 格式为 (x, y)
        cell_radius: 细胞半径（像素）
    
    返回:
        mask: numpy数组 (height, width)，0=背景, 1=阳性细胞, 2=阴性细胞
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 创建PIL图像用于绘制
    mask_img = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask_img)
    
    # 绘制阴性细胞（标签=2）- 先画阴性，这样阳性可以覆盖（如果有重叠）
    for coord in negative_coords:
        x, y = coord
        # 绘制圆形
        draw.ellipse([x-cell_radius, y-cell_radius, x+cell_radius, y+cell_radius], 
                     fill=2, outline=2)
    
    # 绘制阳性细胞（标签=1）
    for coord in positive_coords:
        x, y = coord
        # 绘制圆形
        draw.ellipse([x-cell_radius, y-cell_radius, x+cell_radius, y+cell_radius], 
                     fill=1, outline=1)
    
    mask = np.array(mask_img)
    return mask


def load_coordinates_from_h5(h5_file):
    """从.h5文件加载坐标"""
    if not os.path.exists(h5_file):
        return np.array([]).reshape(0, 2)
    
    with h5py.File(h5_file, 'r') as f:
        if 'coordinates' in f:
            coords = f['coordinates'][...]
            return coords
        else:
            return np.array([]).reshape(0, 2)


def convert_bcdata_to_nnunet(bcdata_root, output_root, dataset_id=100, dataset_name="BCCellSegmentation", 
                             cell_radius=3, include_test=False):
    """
    转换BCData为nnUNetv2格式
    
    参数:
        bcdata_root: BCData数据集根目录路径
        output_root: nnUNet_raw目录路径
        dataset_id: 数据集ID（三位数）
        dataset_name: 数据集名称
        cell_radius: 细胞半径（像素）
        include_test: 是否包含测试集（如果为True，会创建imagesTs但不创建labelsTs）
    """
    bcdata_root = Path(bcdata_root)
    output_root = Path(output_root)
    
    # 创建数据集文件夹
    dataset_folder_name = f"Dataset{dataset_id:03d}_{dataset_name}"
    dataset_path = output_root / dataset_folder_name
    
    print(f"创建数据集目录: {dataset_path}")
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    images_tr_path = dataset_path / "imagesTr"
    labels_tr_path = dataset_path / "labelsTr"
    images_tr_path.mkdir(exist_ok=True)
    labels_tr_path.mkdir(exist_ok=True)
    
    if include_test:
        images_ts_path = dataset_path / "imagesTs"
        images_ts_path.mkdir(exist_ok=True)
    
    # 处理训练集和验证集（合并为训练集）
    splits = ['train', 'validation']
    case_id = 0
    
    for split in splits:
        print(f"\n处理 {split} 数据...")
        
        # 获取该split的所有图像文件
        images_dir = bcdata_root / 'images' / split
        if not images_dir.exists():
            print(f"警告: {images_dir} 不存在，跳过")
            continue
        
        image_files = sorted(list(images_dir.glob('*.png')))
        
        for img_file in tqdm(image_files, desc=f"转换{split}"):
            # 获取基础名称（不含扩展名）
            base_name = img_file.stem
            
            # 读取图像
            img = Image.open(img_file)
            img_array = np.array(img)
            
            # 加载阳性和阴性细胞坐标
            positive_h5 = bcdata_root / 'annotations' / split / 'positive' / f'{base_name}.h5'
            negative_h5 = bcdata_root / 'annotations' / split / 'negative' / f'{base_name}.h5'
            
            positive_coords = load_coordinates_from_h5(positive_h5)
            negative_coords = load_coordinates_from_h5(negative_h5)
            
            # 创建分割掩码
            mask = create_segmentation_mask(
                (img_array.shape[0], img_array.shape[1]),
                positive_coords,
                negative_coords,
                cell_radius=cell_radius
            )
            
            # 生成nnUNet格式的文件名
            case_identifier = f"{dataset_name}_{case_id:04d}"
            
            # 保存图像（nnUNet格式：{CASE_IDENTIFIER}_0000.png）
            # RGB图像可以直接保存为一个文件
            output_image_path = images_tr_path / f"{case_identifier}_0000.png"
            img.save(output_image_path)
            
            # 保存标签（格式：{CASE_IDENTIFIER}.png）
            output_label_path = labels_tr_path / f"{case_identifier}.png"
            mask_img = Image.fromarray(mask)
            mask_img.save(output_label_path)
            
            case_id += 1
    
    num_training_cases = case_id
    print(f"\n训练样本总数: {num_training_cases}")
    
    # 处理测试集（如果需要）
    if include_test:
        print(f"\n处理测试集...")
        test_images_dir = bcdata_root / 'images' / 'test'
        if test_images_dir.exists():
            image_files = sorted(list(test_images_dir.glob('*.png')))
            
            for idx, img_file in enumerate(tqdm(image_files, desc="转换测试集")):
                base_name = img_file.stem
                img = Image.open(img_file)
                
                # 生成测试集文件名
                case_identifier = f"{dataset_name}_test_{idx:04d}"
                output_image_path = images_ts_path / f"{case_identifier}_0000.png"
                img.save(output_image_path)
    
    # 生成dataset.json
    print("\n生成dataset.json...")
    dataset_json = {
        "channel_names": {
            "0": "R",
            "1": "G",
            "2": "B"
        },
        "labels": {
            "background": 0,
            "positive_cell": 1,
            "negative_cell": 2
        },
        "numTraining": num_training_cases,
        "file_ending": ".png",
        "dataset_name": dataset_name,
        "description": "Breast Cancer Cell Segmentation - Positive and Negative Cells"
    }
    
    json_path = dataset_path / "dataset.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n转换完成！")
    print(f"数据集路径: {dataset_path}")
    print(f"训练样本数: {num_training_cases}")
    print(f"类别: 背景(0), 阳性细胞(1), 阴性细胞(2)")
    
    return dataset_path


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='将BCData数据集转换为nnUNetv2格式',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input', 
        type=str, 
        default='./BCData',
        help='BCData数据集根目录路径（包含images和annotations子目录）'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='./nnUNet_raw',
        help='nnUNet_raw目录路径'
    )
    
    parser.add_argument(
        '--dataset_id', 
        type=int, 
        default=100,
        help='数据集ID（三位数，例如100）'
    )
    
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        default='Ki67',
        help='数据集名称'
    )
    
    parser.add_argument(
        '--cell_radius', 
        type=int, 
        default=3,
        help='细胞半径（像素），用于在标注位置绘制圆形'
    )
    
    parser.add_argument(
        '--include_test', 
        action='store_true',
        help='是否包含测试集（会创建imagesTs）'
    )
    
    args = parser.parse_args()
    
    # 检查输入路径
    if not os.path.exists(args.input):
        print(f"错误: 输入路径不存在: {args.input}")
        return
    
    # 转换数据集
    convert_bcdata_to_nnunet(
        bcdata_root=args.input,
        output_root=args.output,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        cell_radius=args.cell_radius,
        include_test=args.include_test
    )


if __name__ == "__main__":
    main()

