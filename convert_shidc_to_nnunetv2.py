#!/usr/bin/env python3
"""
将SHIDC-B-Ki-67数据集转换为nnUNetv2格式
SHIDC-B-Ki-67包含：
- bare images/Train/: 原始JPG图像和对应的JSON点标注文件
- bare images/Test/: 测试JPG图像和对应的JSON点标注文件
- JSON格式: [{"x": int, "y": int, "label_id": int}, ...]
  - label_id: 1=阳性细胞, 2=阴性细胞, 3=TIL

转换为nnUNetv2格式：
- imagesTr/: 训练图像,图像名称为{dataset_name}_{case_id:04d}_0000.png,640x640 RGB格式的png图像
- labelsTr/: 对应的分割标签（0=背景, 1=阳性细胞, 2=阴性细胞, [3=TIL]）
- imagesTs/: 测试图像（如果include_test为True）
- dataset.json: 元数据
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
import argparse
from pathlib import Path
from tqdm import tqdm


def resize_image_and_coordinates(image, coordinates, target_size=(640, 640)):
    """
    调整图像大小并相应调整坐标
    
    参数:
        image: PIL Image对象
        coordinates: 点坐标列表 [{"x": int, "y": int, "label_id": int}, ...]
        target_size: 目标尺寸 (width, height)
    
    返回:
        resized_image: 调整大小后的图像
        resized_coords: 调整后的坐标
    """
    orig_width, orig_height = image.size
    target_width, target_height = target_size
    
    # 计算缩放比例
    scale_x = target_width / orig_width
    scale_y = target_height / orig_height
    
    # 调整图像大小
    resized_image = image.resize(target_size, Image.BILINEAR)
    
    # 调整坐标
    resized_coords = []
    for coord in coordinates:
        new_x = int(coord["x"] * scale_x)
        new_y = int(coord["y"] * scale_y)
        resized_coords.append({
            "x": new_x,
            "y": new_y,
            "label_id": coord["label_id"]
        })
    
    return resized_image, resized_coords


def create_segmentation_mask(image_shape, coordinates, cell_radius=3, include_til=False):
    """
    根据细胞坐标创建分割掩码
    
    参数:
        image_shape: (height, width) 图像尺寸
        coordinates: 点坐标列表 [{"x": int, "y": int, "label_id": int}, ...]
        cell_radius: 细胞半径（像素）
        include_til: 是否包含TIL类别
    
    返回:
        mask: numpy数组 (height, width)
              如果include_til=False: 0=背景, 1=阳性细胞, 2=阴性细胞
              如果include_til=True: 0=背景, 1=阳性细胞, 2=阴性细胞, 3=TIL
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 创建PIL图像用于绘制
    mask_img = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask_img)
    
    # 按label_id分组
    coords_by_label = {1: [], 2: [], 3: []}
    for coord in coordinates:
        label_id = coord["label_id"]
        if label_id in coords_by_label:
            coords_by_label[label_id].append(coord)
    
    # 绘制顺序：先画标签值大的，后画标签值小的（这样小的可以覆盖大的，如果有重叠）
    # 但为了让阳性细胞具有最高优先级，我们按以下顺序绘制
    
    if include_til:
        # 如果包含TIL，绘制顺序: TIL(3) -> 阴性(2) -> 阳性(1)
        # 这样阳性具有最高优先级
        for coord in coords_by_label[3]:
            x, y = coord["x"], coord["y"]
            draw.ellipse([x-cell_radius, y-cell_radius, x+cell_radius, y+cell_radius], 
                        fill=3, outline=3)
    
    # 绘制阴性细胞（标签=2）
    for coord in coords_by_label[2]:
        x, y = coord["x"], coord["y"]
        draw.ellipse([x-cell_radius, y-cell_radius, x+cell_radius, y+cell_radius], 
                     fill=2, outline=2)
    
    # 绘制阳性细胞（标签=1）- 最后绘制，具有最高优先级
    for coord in coords_by_label[1]:
        x, y = coord["x"], coord["y"]
        draw.ellipse([x-cell_radius, y-cell_radius, x+cell_radius, y+cell_radius], 
                     fill=1, outline=1)
    
    mask = np.array(mask_img)
    return mask


def load_json_annotations(json_file):
    """从JSON文件加载点标注"""
    if not os.path.exists(json_file):
        return []
    
    with open(json_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    return annotations


def convert_shidc_to_nnunet(shidc_root, output_root, dataset_id=101, dataset_name="SHIDC_Ki67",
                            cell_radius=3, include_til=False, include_test=True):
    """
    转换SHIDC-B-Ki-67为nnUNetv2格式
    
    参数:
        shidc_root: SHIDC-B-Ki-67数据集根目录路径（包含bare images子目录）
        output_root: nnUNet_raw目录路径
        dataset_id: 数据集ID（三位数）
        dataset_name: 数据集名称
        cell_radius: 细胞半径（像素）
        include_til: 是否包含TIL类别
        include_test: 是否包含测试集
    """
    shidc_root = Path(shidc_root)
    output_root = Path(output_root)
    
    # bare images目录
    bare_images_root = shidc_root / "bare images"
    
    if not bare_images_root.exists():
        print(f"错误: {bare_images_root} 不存在")
        return
    
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
    
    # 处理训练集
    print("\n处理训练集...")
    train_dir = bare_images_root / "Train"
    
    if not train_dir.exists():
        print(f"错误: {train_dir} 不存在")
        return
    
    # 获取所有JPG图像文件
    image_files = sorted(list(train_dir.glob("*.jpg")))
    print(f"找到 {len(image_files)} 个训练图像")
    
    case_id = 0
    for img_file in tqdm(image_files, desc="转换训练集"):
        # 获取基础名称（不含扩展名）
        base_name = img_file.stem
        
        # 对应的JSON文件
        json_file = train_dir / f"{base_name}.json"
        
        # 读取图像
        img = Image.open(img_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 加载标注
        annotations = load_json_annotations(json_file)
        
        # 调整图像大小到640x640，并相应调整坐标
        resized_img, resized_coords = resize_image_and_coordinates(img, annotations, target_size=(640, 640))
        
        # 创建分割掩码
        mask = create_segmentation_mask(
            (640, 640),
            resized_coords,
            cell_radius=cell_radius,
            include_til=include_til
        )
        
        # 生成nnUNet格式的文件名
        case_identifier = f"{dataset_name}_{case_id:04d}"
        
        # 保存图像（nnUNet格式：{CASE_IDENTIFIER}_0000.png）
        output_image_path = images_tr_path / f"{case_identifier}_0000.png"
        resized_img.save(output_image_path)
        
        # 保存标签（格式：{CASE_IDENTIFIER}.png）
        output_label_path = labels_tr_path / f"{case_identifier}.png"
        mask_img = Image.fromarray(mask)
        mask_img.save(output_label_path)
        
        case_id += 1
    
    num_training_cases = case_id
    print(f"\n训练样本总数: {num_training_cases}")
    
    # 处理测试集（如果需要）
    if include_test:
        print("\n处理测试集...")
        test_dir = bare_images_root / "Test"
        
        if test_dir.exists():
            test_image_files = sorted(list(test_dir.glob("*.jpg")))
            print(f"找到 {len(test_image_files)} 个测试图像")
            
            for idx, img_file in enumerate(tqdm(test_image_files, desc="转换测试集")):
                # 读取图像
                img = Image.open(img_file)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 调整图像大小到640x640
                resized_img = img.resize((640, 640), Image.BILINEAR)
                
                # 生成测试集文件名
                case_identifier = f"{dataset_name}_test_{idx:04d}"
                output_image_path = images_ts_path / f"{case_identifier}_0000.png"
                resized_img.save(output_image_path)
        else:
            print(f"警告: {test_dir} 不存在，跳过测试集")
    
    # 生成dataset.json
    print("\n生成dataset.json...")
    
    if include_til:
        labels_dict = {
            "background": 0,
            "positive_cell": 1,
            "negative_cell": 2,
            "TIL": 3
        }
        description = "SHIDC-B-Ki-67 Cell Segmentation - Positive, Negative Cells and TIL"
    else:
        labels_dict = {
            "background": 0,
            "positive_cell": 1,
            "negative_cell": 2
        }
        description = "SHIDC-B-Ki-67 Cell Segmentation - Positive and Negative Cells"
    
    dataset_json = {
        "channel_names": {
            "0": "R",
            "1": "G",
            "2": "B"
        },
        "labels": labels_dict,
        "numTraining": num_training_cases,
        "file_ending": ".png",
        "dataset_name": dataset_name,
        "description": description
    }
    
    json_path = dataset_path / "dataset.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n转换完成！")
    print(f"数据集路径: {dataset_path}")
    print(f"训练样本数: {num_training_cases}")
    if include_til:
        print(f"类别: 背景(0), 阳性细胞(1), 阴性细胞(2), TIL(3)")
    else:
        print(f"类别: 背景(0), 阳性细胞(1), 阴性细胞(2)")
    print(f"图像尺寸: 640x640")
    
    return dataset_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='将SHIDC-B-Ki-67数据集转换为nnUNetv2格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本转换（忽略TIL）
  python convert_shidc_to_nnunetv2.py --input ./SHIDC-B-Ki-67 --output ./nnUNet_raw
  
  # 包含TIL类别
  python convert_shidc_to_nnunetv2.py --input ./SHIDC-B-Ki-67 --output ./nnUNet_raw --include_til
  
  # 自定义参数
  python convert_shidc_to_nnunetv2.py --input ./SHIDC-B-Ki-67 --output ./nnUNet_raw \\
      --dataset_id 101 --dataset_name SHIDC_Ki67 --cell_radius 5 --include_til
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='./SHIDC-B-Ki-67',
        help='SHIDC-B-Ki-67数据集根目录路径（包含bare images子目录）'
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
        default=101,
        help='数据集ID（三位数，例如101）'
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
        '--include_til',
        action='store_true',
        help='是否包含TIL类别（默认忽略TIL，只保留阳性和阴性细胞）'
    )
    
    parser.add_argument(
        '--include_test',
        action='store_true',
        default=True,
        help='是否包含测试集（会创建imagesTs）'
    )
    
    parser.add_argument(
        '--no_test',
        dest='include_test',
        action='store_false',
        help='不包含测试集'
    )
    
    args = parser.parse_args()
    
    # 检查输入路径
    if not os.path.exists(args.input):
        print(f"错误: 输入路径不存在: {args.input}")
        return
    
    # 转换数据集
    convert_shidc_to_nnunet(
        shidc_root=args.input,
        output_root=args.output,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        cell_radius=args.cell_radius,
        include_til=args.include_til,
        include_test=args.include_test
    )


if __name__ == "__main__":
    main()

