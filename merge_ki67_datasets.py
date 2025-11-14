#!/usr/bin/env python3
"""
合并多个 nnUNet v2 数据集到一个新的 DatasetXXX 目录。

示例：
    python3 merge_ki67_datasets.py \
        --source-datasets Dataset100_Ki67 Dataset101_Ki67 \
        --target-id 150 \
        --target-name Ki67Combined

默认会从 nnUNet_raw（取自环境变量或当前仓库根目录下的同名文件夹）读取源数据集，
将训练与测试图像/标签复制到新的数据集中，并生成新的 dataset.json。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并多个 nnUNet v2 数据集")
    parser.add_argument(
        "--source-datasets",
        "-s",
        nargs="+",
        required=True,
        help="待合并的数据集目录名，例如 Dataset100_Ki67 Dataset101_Ki67。",
    )
    parser.add_argument(
        "--target-id",
        type=int,
        required=True,
        help="新数据集的三位 ID（整数），脚本会格式化为 DatasetXXX_...",
    )
    parser.add_argument(
        "--target-name",
        required=True,
        help="新数据集目录名中的名称部分，例如 Ki67Combined。",
    )
    parser.add_argument(
        "--nnunet-raw",
        default=os.environ.get("nnUNet_raw"),
        help="nnUNet_raw 根目录。默认取环境变量 nnUNet_raw，如未设置则使用当前仓库根目录下的 nnUNet_raw。",
    )
    parser.add_argument(
        "--train-prefix",
        default="Ki67Combined",
        help="合并后训练集病例名前缀（默认 Ki67Combined）。",
    )
    parser.add_argument(
        "--test-prefix",
        default=None,
        help="合并后测试集病例名前缀（默认为 train-prefix + 'Ts'）。",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="dataset.json 中的 description 字段，默认自动生成。",
    )
    parser.add_argument(
        "--converted-by",
        default="Dataset merger script",
        help="dataset.json 中的 converted_by 字段。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="目标目录已存在时覆盖其内容（谨慎使用，会删除现有内容）。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要执行的操作，不实际复制文件或写入 dataset.json。",
    )
    return parser.parse_args()


def resolve_raw_base(raw_arg: str | None) -> Path:
    if raw_arg:
        raw_path = Path(raw_arg).expanduser().resolve()
    else:
        raw_path = Path(__file__).resolve().parent / "nnUNet_raw"
    if not raw_path.exists():
        raise FileNotFoundError(f"nnUNet_raw 根目录不存在：{raw_path}")
    return raw_path


def resolve_dataset_dir(raw_base: Path, dataset: str) -> Path:
    candidate = raw_base / dataset
    if candidate.exists():
        return candidate
    if dataset.isdigit():
        matches = list(raw_base.glob(f"Dataset{int(dataset):03d}_*"))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(f"未找到 Dataset{int(dataset):03d}_* 目录")
        raise ValueError(
            f"存在多个 Dataset{int(dataset):03d}_* 目录，请提供完整目录名：{matches}"
        )
    raise FileNotFoundError(f"未找到数据集目录：{candidate}")


def load_dataset_metadata(dataset_dir: Path) -> Dict:
    dataset_json = dataset_dir / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(f"缺少 dataset.json：{dataset_json}")
    return json_load(dataset_json)


def json_load(path: Path) -> Dict:
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_same_metadata(
    metadatas: Iterable[Dict], key: str, description: str
) -> Dict:
    iterator = iter(metadatas)
    first = next(iterator)
    reference = first[key]
    for meta in iterator:
        if meta[key] != reference:
            raise ValueError(
                f"{description} 不一致：{reference} vs {meta[key]}"
            )
    return reference


def gather_channels(images_dir: Path, file_ending: str) -> Dict[str, List[Path]]:
    channel_map: Dict[str, List[Path]] = defaultdict(list)
    for file in sorted(images_dir.glob(f"*{file_ending}")):
        stem = file.name[: -len(file_ending)]
        if "_" not in stem:
            raise ValueError(f"文件命名不符合规范：{file}")
        base, channel = stem.rsplit("_", 1)
        if not channel.isdigit() or len(channel) != 4:
            raise ValueError(f"文件命名不符合 *_XXXX{file_ending} 规则：{file}")
        channel_map[base].append(file)
    return channel_map


def copy_training_cases(
    source_dir: Path,
    dest_images: Path,
    dest_labels: Path,
    start_index: int,
    prefix: str,
    file_ending: str,
    dry_run: bool,
) -> Tuple[int, List[str]]:
    images_tr = source_dir / "imagesTr"
    labels_tr = source_dir / "labelsTr"
    if not images_tr.exists() or not labels_tr.exists():
        raise FileNotFoundError(f"{source_dir} 缺少 imagesTr 或 labelsTr")

    channel_map = gather_channels(images_tr, file_ending)
    copied_cases: List[str] = []
    case_index = start_index

    for case_id in sorted(channel_map.keys()):
        new_case_name = f"{prefix}_{case_index:04d}"
        case_index += 1
        copied_cases.append(new_case_name)

        for channel_path in channel_map[case_id]:
            suffix = channel_path.stem.rsplit("_", 1)[1]
            target_path = dest_images / f"{new_case_name}_{suffix}{file_ending}"
            if dry_run:
                print(f"[DRY-RUN] copy {channel_path} -> {target_path}")
            else:
                shutil.copy2(channel_path, target_path)

        label_src = labels_tr / f"{case_id}{file_ending}"
        if not label_src.exists():
            raise FileNotFoundError(f"缺少标签：{label_src}")
        label_dest = dest_labels / f"{new_case_name}{file_ending}"
        if dry_run:
            print(f"[DRY-RUN] copy {label_src} -> {label_dest}")
        else:
            shutil.copy2(label_src, label_dest)

    return case_index, copied_cases


def copy_test_cases(
    source_dir: Path,
    dest_images: Path,
    start_index: int,
    prefix: str,
    file_ending: str,
    dry_run: bool,
) -> Tuple[int, List[str]]:
    images_ts = source_dir / "imagesTs"
    if not images_ts.exists():
        return start_index, []

    channel_map = gather_channels(images_ts, file_ending)
    copied_cases: List[str] = []
    case_index = start_index

    for case_id in sorted(channel_map.keys()):
        new_case_name = f"{prefix}_{case_index:04d}"
        case_index += 1
        copied_cases.append(new_case_name)

        for channel_path in channel_map[case_id]:
            suffix = channel_path.stem.rsplit("_", 1)[1]
            target_path = dest_images / f"{new_case_name}_{suffix}{file_ending}"
            if dry_run:
                print(f"[DRY-RUN] copy {channel_path} -> {target_path}")
            else:
                shutil.copy2(channel_path, target_path)

    return case_index, copied_cases


def prepare_target_dir(target_dir: Path, overwrite: bool, dry_run: bool) -> None:
    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"目标目录已存在：{target_dir}。使用 --overwrite 可覆盖。"
            )
        if dry_run:
            print(f"[DRY-RUN] would remove existing directory {target_dir}")
        else:
            shutil.rmtree(target_dir)
    if dry_run:
        print(f"[DRY-RUN] would create {target_dir}/{{imagesTr,labelsTr,imagesTs}}")
    else:
        (target_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
        (target_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
        (target_dir / "imagesTs").mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    raw_base = resolve_raw_base(args.nnunet_raw)

    source_dirs = [resolve_dataset_dir(raw_base, ds) for ds in args.source_datasets]
    metadata_list = [load_dataset_metadata(ds) for ds in source_dirs]

    channel_names = ensure_same_metadata(metadata_list, "channel_names", "channel_names")
    labels = ensure_same_metadata(metadata_list, "labels", "labels")
    file_ending = ensure_same_metadata(metadata_list, "file_ending", "file_ending")

    target_dir = raw_base / f"Dataset{args.target_id:03d}_{args.target_name}"
    prepare_target_dir(target_dir, args.overwrite, args.dry_run)

    dest_images_tr = target_dir / "imagesTr"
    dest_labels_tr = target_dir / "labelsTr"
    dest_images_ts = target_dir / "imagesTs"

    train_prefix = args.train_prefix
    test_prefix = args.test_prefix or f"{train_prefix}Ts"

    train_index = 0
    test_index = 0
    all_train_cases: List[str] = []
    all_test_cases: List[str] = []

    for source_dir in source_dirs:
        print(f"处理数据集：{source_dir}")
        train_index, copied_train = copy_training_cases(
            source_dir,
            dest_images_tr,
            dest_labels_tr,
            train_index,
            train_prefix,
            file_ending,
            args.dry_run,
        )
        all_train_cases.extend(copied_train)

        test_index, copied_test = copy_test_cases(
            source_dir,
            dest_images_ts,
            test_index,
            test_prefix,
            file_ending,
            args.dry_run,
        )
        all_test_cases.extend(copied_test)

    if args.dry_run:
        print(
            f"[DRY-RUN] 将生成 dataset.json，numTraining={len(all_train_cases)}, "
            f"train_prefix={train_prefix}, test_prefix={test_prefix}"
        )
        return

    description = (
        args.description
        if args.description is not None
        else f"Combined dataset from {', '.join(ds.name for ds in source_dirs)}"
    )

    channel_names_int_keys = {int(k): v for k, v in channel_names.items()}

    generate_dataset_json(
        output_folder=str(target_dir),
        channel_names=channel_names_int_keys,
        labels=labels,
        num_training_cases=len(all_train_cases),
        file_ending=file_ending,
        dataset_name=args.target_name,
        description=description,
        reference=", ".join(meta.get("dataset_name", ds.name) for meta, ds in zip(metadata_list, source_dirs)),
        release=None,
        converted_by=args.converted_by,
    )

    print(f"合并完成：{target_dir}")
    print(f"训练集病例数：{len(all_train_cases)}，测试集病例数：{len(all_test_cases)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        sys.exit(1)

