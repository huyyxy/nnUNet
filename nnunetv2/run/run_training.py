"""
本文件负责封装 nnU-Net v2 的训练入口。我们提供了单卡与多卡（DDP）两种执行方式，并在这里处理命令行参数解析、
训练器（Trainer）实例化、断点续训、验证流程等工作。为了帮助初学者理解代码逻辑，已经在关键步骤加入教学级别的中文注释。
"""

import multiprocessing
import os
import socket
from typing import Union, Optional

import nnunetv2
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing as mp
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from torch.backends import cudnn


def find_free_network_port() -> int:
    """查找一个当前未被占用的本地端口，用于 DDP 训练中的通信配置。"""
    # 创建一个 TCP 套接字
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 传入端口 0，操作系统会自动分配一个空闲端口
    s.bind(("", 0))
    # 获取操作系统实际分配的端口号
    port = s.getsockname()[1]
    # 记得释放资源，避免端口被长期占用
    s.close()
    return port


def get_trainer_from_args(dataset_name_or_id: Union[int, str],
                          configuration: str,
                          fold: int,
                          trainer_name: str = 'nnUNetTrainer',
                          plans_identifier: str = 'nnUNetPlans',
                          device: torch.device = torch.device('cuda')):
    """
    根据命令行参数构建训练器（Trainer）实例。

    参数说明（为初学者准备）：
    - dataset_name_or_id: 数据集的名称（如 Dataset123_Example）或数字 ID。
    - configuration: nnU-Net 的配置名，决定网络结构、预处理等。
    - fold: 5 折交叉验证中的折编号。
    - trainer_name: 指定 Trainer 类名，可自定义。
    - plans_identifier: 指定使用的 plans 文件前缀，默认 nnUNetPlans。
    - device: 指定要运行的设备（cuda/cpu/mps）。
    """
    # 第一步：根据字符串名称动态查找 Trainer 类
    nnunet_trainer = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                trainer_name, 'nnunetv2.training.nnUNetTrainer')
    if nnunet_trainer is None:
        raise RuntimeError(
            f'未找到名为 {trainer_name} 的 Trainer 类。请确认它位于 '
            f'nnunetv2.training.nnUNetTrainer 目录（路径：'
            f'{join(nnunetv2.__path__[0], "training", "nnUNetTrainer")}），或将文件移动至该目录。'
        )
    assert issubclass(nnunet_trainer, nnUNetTrainer), '指定的 Trainer 类必须继承自 nnUNetTrainer，请确认继承关系。'

    # 处理 dataset_name_or_id。用户可能传入 DatasetXXX_name 或纯数字，此处逐一兼容。
    if dataset_name_or_id.startswith('Dataset'):
        pass
    else:
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError(
                f'dataset_name_or_id 只能是整数 ID，或形如 DatasetXXX_YYY 的数据集名称（XXX 必须是三位任务编号）。'
                f'当前输入：{dataset_name_or_id}'
            )

    # 利用配置文件（plans.json）构建 Trainer 实例，这些文件是在数据预处理阶段生成的
    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + '.json')
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, 'dataset.json'))
    nnunet_trainer = nnunet_trainer(plans=plans, configuration=configuration, fold=fold,
                                    dataset_json=dataset_json, device=device)
    return nnunet_trainer


def maybe_load_checkpoint(nnunet_trainer: nnUNetTrainer, continue_training: bool, validation_only: bool,
                          pretrained_weights_file: str = None):
    """
    根据用户意图选择加载不同的权重文件。

    场景说明：
    - continue_training=True：从已有训练断点继续，需要找到最新的 checkpoint。
    - validation_only=True：仅做验证，要求训练已经完成（有 checkpoint_final）。
    - pretrained_weights_file：提前加载外部预训练权重，只适用于全新训练的起点。
    """
    if continue_training and pretrained_weights_file is not None:
        raise RuntimeError('不能同时开启断点续训 (--c) 和加载预训练权重，预训练权重仅适用于全新训练的起点。')
    if continue_training:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_latest.pth')
        # special case where --c is used to run a previously aborted validation
        if not isfile(expected_checkpoint_file):
            expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_best.pth')
        if not isfile(expected_checkpoint_file):
            print("警告：未找到可用于继续训练的 checkpoint，将从头开始新的训练流程。")
            expected_checkpoint_file = None
    elif validation_only:
        expected_checkpoint_file = join(nnunet_trainer.output_folder, 'checkpoint_final.pth')
        if not isfile(expected_checkpoint_file):
            raise RuntimeError("无法执行验证：尚未完成训练，缺少 checkpoint_final.pth。")
    else:
        if pretrained_weights_file is not None:
            if not nnunet_trainer.was_initialized:
                nnunet_trainer.initialize()
            load_pretrained_weights(nnunet_trainer.network, pretrained_weights_file, verbose=True)
        expected_checkpoint_file = None

    if expected_checkpoint_file is not None:
        nnunet_trainer.load_checkpoint(expected_checkpoint_file)


def setup_ddp(rank, world_size):
    """初始化分布式训练所需的进程组。"""
    # backend 选择 nccl（针对 NVIDIA GPU 优化），rank 是当前进程编号，world_size 是总进程数
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    """销毁进程组，释放分布式环境占用的资源。"""
    dist.destroy_process_group()


def run_ddp(rank, dataset_name_or_id, configuration, fold, tr, p, disable_checkpointing, c, val,
            pretrained_weights, npz, val_with_best, world_size):
    """
    单个子进程在 DDP 环境下执行的核心逻辑。

    参数中 rank/world_size 由 mp.spawn 传入，其余参数等价于 run_training 的对应选项。
    """
    setup_ddp(rank, world_size)
    # 每个进程绑定到不同的 GPU（默认 rank 与 GPU ID 对齐）
    torch.cuda.set_device(torch.device('cuda', dist.get_rank()))

    nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, tr, p)

    if disable_checkpointing:
        nnunet_trainer.disable_checkpointing = disable_checkpointing

    assert not (c and val), '--c 与 --val 不能同时启用，请只选择其中一个。'

    maybe_load_checkpoint(nnunet_trainer, c, val, pretrained_weights)

    if torch.cuda.is_available():
        # 禁用确定性模式，启用 benchmark 可以让 cudnn 为当前输入规模寻找最优算法
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not val:
        # 若不是只执行验证，则开始常规训练循环
        nnunet_trainer.run_training()

    if val_with_best:
        nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
    nnunet_trainer.perform_actual_validation(npz)
    cleanup_ddp()


def run_training(dataset_name_or_id: Union[str, int],
                 configuration: str, fold: Union[int, str],
                 trainer_class_name: str = 'nnUNetTrainer',
                 plans_identifier: str = 'nnUNetPlans',
                 pretrained_weights: Optional[str] = None,
                 num_gpus: int = 1,
                 export_validation_probabilities: bool = False,
                 continue_training: bool = False,
                 only_run_validation: bool = False,
                 disable_checkpointing: bool = False,
                 val_with_best: bool = False,
                 device: torch.device = torch.device('cuda')):
    """
    公开给外部调用的主训练函数。这里根据 num_gpus 决定走单卡逻辑还是多卡 DDP。

    建议阅读流程：
    1. 参数检查（折号、设备类型等）。
    2. DDP 条件分支：多卡使用 mp.spawn 启动多个进程。
    3. 单卡流程：构造 Trainer、加载权重、训练与验证。
    """
    if plans_identifier == 'nnUNetPlans':
        print("\n############################\n"
              "提示：当前使用的是旧版 nnU-Net 默认 plans。我们已经提供了更新的推荐配置，"
              "请考虑改用新方案。详情请查看：https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md"
              "\n############################\n")
    if isinstance(fold, str):
        if fold != 'all':
            try:
                fold = int(fold)
            except ValueError as e:
                print(f'无法将 fold 转为整数：{fold}。fold 只能取 "all" 或 0-4 之间的整数。')
                raise e

    if val_with_best:
        assert not disable_checkpointing, '--val_best 与 --disable_checkpointing 不兼容，请取消其中一个选项。'

    if num_gpus > 1:
        assert device.type == 'cuda', f"当 num_gpus 大于 1 时仅支持 CUDA 设备，当前设备为 {device}，请改用 cuda。"

        # DDP 训练需要设定主进程的地址与端口，这里默认使用本机
        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ.keys():
            port = str(find_free_network_port())
            print(f"已为 DDP 通信自动分配端口 {port}")
            print("正在使用自动分配的 DDP 端口，请保持本机端口未被其他进程占用。")
            os.environ['MASTER_PORT'] = port  # str(port)

        # 使用 torch.multiprocessing.spawn 启动多个子进程，每个进程调用 run_ddp
        mp.spawn(run_ddp,
                 args=(
                     dataset_name_or_id,
                     configuration,
                     fold,
                     trainer_class_name,
                     plans_identifier,
                     disable_checkpointing,
                     continue_training,
                     only_run_validation,
                     pretrained_weights,
                     export_validation_probabilities,
                     val_with_best,
                     num_gpus),
                 nprocs=num_gpus,
                 join=True)
    else:
        # 单 GPU 或 CPU 等情况直接在当前进程中运行
        nnunet_trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_class_name,
                                               plans_identifier, device=device)

        if disable_checkpointing:
            nnunet_trainer.disable_checkpointing = disable_checkpointing

        assert not (continue_training and only_run_validation), '--c 与 --val 不能同时启用，请只选择其中一个。'

        maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation, pretrained_weights)

        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        if not only_run_validation:
            # 真正的训练主循环在 Trainer 内部实现，这里只负责触发
            nnunet_trainer.run_training()

        if val_with_best:
            nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, 'checkpoint_best.pth'))
        nnunet_trainer.perform_actual_validation(export_validation_probabilities)


def run_training_entry():
    """
    命令行入口函数。

    处理命令行参数解析，并将参数传给上面的 run_training。
    大多数用户直接调用 `nnUNetv2_train` 时最终会执行到这里。
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name_or_id', type=str,
                        help="要训练的数据集名称或 ID，例如 Dataset123_Example 或 123。")
    parser.add_argument('configuration', type=str,
                        help="要使用的网络配置名称，对应 nnU-Net 预设的配置文件。")
    parser.add_argument('fold', type=str,
                        help='5 折交叉验证中的折编号，可选 0-4，或输入 all 表示全部折。')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='【可选】指定自定义 Trainer 类名，默认 nnUNetTrainer。')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='【可选】指定自定义的 plans 标识符，默认 nnUNetPlans。')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='【可选】预训练模型 checkpoint 路径，仅在开始训练时生效，勿与断点续训同时使用。')
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='设置训练时使用的 GPU 数量，大于 1 时自动启用 DDP。')
    parser.add_argument('--npz', action='store_true', required=False,
                        help='【可选】在最终验证时额外保存 softmax 预测为 npz 文件，便于后续模型集成分析。')
    parser.add_argument('--c', action='store_true', required=False,
                        help='【可选】从最近的 checkpoint 继续训练。')
    parser.add_argument('--val', action='store_true', required=False,
                        help='【可选】仅运行验证（要求已有训练完成的 checkpoint_final）。')
    parser.add_argument('--val_best', action='store_true', required=False,
                        help='【可选】使用 checkpoint_best 进行验证，不能与 --disable_checkpointing 同时使用，并会覆盖原验证输出。')
    parser.add_argument('--disable_checkpointing', action='store_true', required=False,
                        help='【可选】禁用训练过程中的 checkpoint 保存，适合做快速实验以避免占用硬盘空间。')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                    help="设置训练所使用的设备，可选 'cuda'（GPU）、'cpu'、'mps'（苹果芯片）。如需指定 GPU ID，请使用 CUDA_VISIBLE_DEVICES。")
    args = parser.parse_args()

    assert args.device in ['cpu', 'cuda', 'mps'], f'-device 只能取 cpu、cuda 或 mps，当前输入为 {args.device}，请重新检查。'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    run_training(args.dataset_name_or_id, args.configuration, args.fold, args.tr, args.p, args.pretrained_weights,
                 args.num_gpus, args.npz, args.c, args.val, args.disable_checkpointing, args.val_best,
                 device=device)


if __name__ == '__main__':
    # 设置若干环境变量，限制底层数学库的线程数，避免资源争用
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # reduces the number of threads used for compiling. More threads don't help and can cause problems
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = 1
    # multiprocessing.set_start_method("spawn")
    run_training_entry()
