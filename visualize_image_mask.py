#!/usr/bin/env python3
"""
根据原始图像与掩码图像可视化阳性/阴性细胞信息。

该脚本复现 ``visualize_bcdata.py`` 的可视化布局，只是将输入改为：
  * 原始图像（PNG/JPG 等）
  * 掩码图像（同尺寸，像素取值或颜色区分类别）

默认假设掩码像素值为：
  * 2 -> 阳性 (positive)
  * 1 -> 阴性 (negative)

如果你的掩码使用其它取值或颜色，可通过命令行参数进行配置。
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


Color = Tuple[int, int, int]
Point = Tuple[float, float]


def parse_value_tokens(tokens: Sequence[str]) -> List[int]:
    """
    将用户输入的取值列表解析为整数列表。

    支持的格式：
      * 十进制整数，如 ``2``、``10``
      * 带 0x 前缀的十六进制，如 ``0x02``
      * 颜色字符串 ``#RRGGBB``，会被编码为 ``R<<16 | G<<8 | B``
    """
    result: List[int] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token.startswith("#"):
            if len(token) != 7:
                raise ValueError(f"颜色值需要形如 #RRGGBB：{token}")
            r = int(token[1:3], 16)
            g = int(token[3:5], 16)
            b = int(token[5:7], 16)
            encoded = (r << 16) | (g << 8) | b
            result.append(encoded)
        else:
            result.append(int(token, 0))
    return result


def load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    return Image.open(path).convert("RGB")


def load_mask_as_labels(path: Path) -> np.ndarray:
    """
    将掩码图像转换为二维整型数组。

    若掩码为灰度/单通道，直接返回像素值。
    若为 RGB(A)，将 (R,G,B) 编码为 24 位整数方便比较。
    """
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    mask_img = Image.open(path)
    mask_arr = np.array(mask_img)
    if mask_arr.ndim == 2:
        return mask_arr.astype(np.int32)
    if mask_arr.ndim == 3:
        if mask_arr.shape[2] not in (3, 4):
            raise ValueError(f"暂不支持形状为 {mask_arr.shape} 的掩码图像")
        rgb = mask_arr[..., :3].astype(np.int32)
        encoded = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
        return encoded
    raise ValueError(f"无法解析掩码图像，形状：{mask_arr.shape}")


def extract_component_centers(
    labels: np.ndarray, target_values: Iterable[int], connectivity: int = 8
) -> List[Point]:
    """
    从标签图中提取指定取值的连通域中心坐标。

    Parameters
    ----------
    labels:
        二维整型数组。
    target_values:
        需要提取的像素取值（可以包含多个类别值）。
    connectivity:
        4 或 8，表示连通性。
    """
    if labels.ndim != 2:
        raise ValueError("labels 必须是二维数组")
    if connectivity not in (4, 8):
        raise ValueError("connectivity 仅支持 4 或 8")

    target_set = set(int(v) for v in target_values)
    if not target_set:
        return []

    height, width = labels.shape
    visited = np.zeros_like(labels, dtype=bool)
    centers: List[Point] = []

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    for y in range(height):
        for x in range(width):
            if visited[y, x]:
                continue
            current_value = int(labels[y, x])
            if current_value not in target_set:
                continue

            stack = deque([(y, x)])
            visited[y, x] = True
            pixels: List[Tuple[int, int]] = []

            while stack:
                cy, cx = stack.pop()
                pixels.append((cx, cy))
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not visited[ny, nx]
                        and int(labels[ny, nx]) == current_value
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))

            if pixels:
                xs, ys = zip(*pixels)
                centers.append((float(np.mean(xs)), float(np.mean(ys))))

    return centers


def draw_points(
    base_image: Image.Image,
    points: Sequence[Point],
    outline_color: Color,
    fill: bool,
    radius: int,
) -> Image.Image:
    result = base_image.copy()
    draw = ImageDraw.Draw(result)
    for px, py in points:
        x, y = int(round(px)), int(round(py))
        bbox = [
            (x - radius, y - radius),
            (x + radius, y + radius),
        ]
        if fill:
            draw.ellipse(bbox, fill=outline_color, outline=outline_color)
        else:
            draw.ellipse(bbox, outline=outline_color, width=2)
    return result


def visualize(
    image_path: Path,
    mask_path: Path,
    positive_values: Sequence[int],
    negative_values: Sequence[int],
    point_radius: int,
    connectivity: int,
    output_path: Path | None,
    show: bool,
) -> Path:
    image = load_image(image_path)
    labels = load_mask_as_labels(mask_path)

    positive_points = extract_component_centers(
        labels, positive_values, connectivity
    )
    negative_points = extract_component_centers(
        labels, negative_values, connectivity
    )

    num_positive = len(positive_points)
    num_negative = len(negative_points)
    total = num_positive + num_negative
    ki67_index = (num_positive / total * 100.0) if total > 0 else 0.0

    print(f"原始图像: {image_path}")
    print(f"掩码图像: {mask_path}")
    print(f"图像尺寸: {image.size[1]} x {image.size[0]}")
    print("\n类别统计:")
    print(f"  阳性细胞 (Positive): {num_positive}")
    print(f"  阴性细胞 (Negative): {num_negative}")
    print(f"  总计: {total}")
    if total > 0:
        print(f"  Ki-67增殖指数: {ki67_index:.2f}%")

    positive_color: Color = (255, 0, 0)
    negative_color: Color = (0, 0, 255)

    overlay = draw_points(image, negative_points, negative_color, False, point_radius)
    overlay = draw_points(overlay, positive_points, positive_color, False, point_radius)

    points_only = Image.new("RGB", image.size, (0, 0, 0))
    points_only = draw_points(
        points_only, negative_points, negative_color, True, point_radius
    )
    points_only = draw_points(
        points_only, positive_points, positive_color, True, point_radius
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    title_text = "Annotated Points\n"
    title_text += f"Red={num_positive} (Positive)\n"
    title_text += f"Blue={num_negative} (Negative)\n"
    if total > 0:
        title_text += f"Ki-67 Index={ki67_index:.1f}%"
    axes[1].imshow(points_only)
    axes[1].set_title(title_text, fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay Display\n(Red=Positive, Blue=Negative)", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()

    if output_path is None:
        output_path = image_path.with_name(f"visualization_mask_{image_path.stem}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n可视化结果已保存到: {output_path}")

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="根据原图与掩码生成阳性/阴性可视化结果",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, type=Path, help="原始图像路径")
    parser.add_argument("--mask", required=True, type=Path, help="掩码图像路径")
    parser.add_argument(
        "--positive-values",
        nargs="+",
        default=["1"],
        help="掩码中表示阳性细胞的像素取值或颜色（支持 #RRGGBB）",
    )
    parser.add_argument(
        "--negative-values",
        nargs="+",
        default=["2"],
        help="掩码中表示阴性细胞的像素取值或颜色（支持 #RRGGBB）",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=6,
        help="绘制圆点的半径",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        default=8,
        choices=[4, 8],
        help="计算连通域所使用的连通性",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出图像路径（默认与原图同目录）",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="处理完毕后直接显示图像（需要 GUI 支持）",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    positive_values = parse_value_tokens(args.positive_values)
    negative_values = parse_value_tokens(args.negative_values)

    if not positive_values:
        raise ValueError("必须至少指定一个阳性像素取值")
    if not negative_values:
        raise ValueError("必须至少指定一个阴性像素取值")

    visualize(
        image_path=args.image,
        mask_path=args.mask,
        positive_values=positive_values,
        negative_values=negative_values,
        point_radius=args.radius,
        connectivity=args.connectivity,
        output_path=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()

