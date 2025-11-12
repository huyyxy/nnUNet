# 系统要求

## 操作系统
nnU-Net 已在 Linux（Ubuntu 18.04、20.04、22.04；centOS、RHEL）、Windows 和 MacOS 上完成测试，开箱即用！

## 硬件要求
我们支持 GPU（推荐）、CPU 以及 Apple M1/M2 设备（目前 Apple mps 尚未实现 3D 卷积，因此在这些设备上可能需要使用 CPU）。

### 训练的硬件要求
我们推荐使用 GPU 进行训练，因为在 CPU 或 MPS（Apple M1/M2）上训练所需时间会非常长。训练时需要至少 10 GB 显存的 GPU（常见的非数据中心选项包括 RTX 2080 Ti、RTX 3080/3090 或 RTX 4080/4090）。我们也建议配备一颗性能强劲的 CPU 来配合 GPU。最低配置为 6 核（12 线程）！CPU 的要求主要与数据增强有关，并且会随着输入通道数和目标结构数量的增加而提升。此外，GPU 越快，CPU 也应该越强！

### 推理的硬件要求
我们同样推荐使用 GPU 进行推理，因为这会比其他选项快得多。不过，CPU 和 MPS（Apple M1/M2）上的推理时间也通常在可接受范围内。如果使用 GPU，需保证至少有 4 GB 可用（未占用）的显存。

### 硬件配置示例
训练用工作站示例：
- CPU：Ryzen 5800X；5900X 或 7900X 会更好！我们尚未测试 Intel Alder/Raptor Lake，但它们很可能同样适用。
- GPU：RTX 3090 或 RTX 4090
- 内存：64 GB
- 存储：SSD（M.2 PCIe Gen 3 或更高规格）

训练用服务器示例：
- CPU：2 × AMD EPYC 7763，共计 128C/256T。对于 A100 等高速 GPU，强烈建议每块 GPU 至少配备 16 个 CPU 核心！
- GPU：8 × A100 PCIe（相较 SXM 版本具备更高的性价比且耗电更少）
- 内存：1 TB
- 存储：本地 SSD（PCIe Gen 3 或更高）或超高速网络存储

（nnU-Net 默认每次训练使用一块 GPU。上述服务器配置最多可同时运行 8 个模型训练。）

### 正确设置数据增强的工作进程数量（仅训练）
请根据 CPU/GPU 的比例手动设置 nnU-Net 在数据增强时使用的进程数量。对于上面的服务器（8 块 GPU 共 256 线程），推荐值在 24-30 之间。可通过设置环境变量 `nnUNet_n_proc_DA` 来实现（`export nnUNet_n_proc_DA=XX`）。推荐值（假设使用具备优异 IPC 的新款 CPU）为：RTX 2080 Ti 使用 10-12，RTX 3090 使用 12，RTX 4090 使用 16-18，A100 使用 28-32。最佳数值可能会因输入通道/模态数量及类别数量而变化。

# 安装指南
我们强烈建议在虚拟环境中安装 nnU-Net！无论是 pip 还是 anaconda 都可以。如果选择从源码编译 PyTorch（见下文），则需要使用 conda 而非 pip。

请使用较新的 Python 版本！3.9 或更新版本可以确保正常运行！

**nnU-Net v2 可以与 nnU-Net v1 共存！两者可以同时安装。**

1) 按照官网说明（conda/pip）安装 [PyTorch](https://pytorch.org/get-started/locally/)。请选择与您的硬件（cuda、mps、cpu）相匹配的最新版本。
**在没有正确安装 PyTorch 前，不要直接执行 `pip install nnunetv2`！** 为了追求极致性能，可考虑[自行编译 PyTorch](https://github.com/pytorch/pytorch#from-source)（仅推荐给有经验的用户）。
2) 根据使用场景安装 nnU-Net：
    1) 作为**标准化基线**、**开箱即用的分割算法**或用于运行**预训练模型的推理**：

       ```pip install nnunetv2```

    2) 作为集成化**框架**使用（这会在本地创建 nnU-Net 的代码副本，以便您按需修改）：
          ```bash
          git clone https://github.com/MIC-DKFZ/nnUNet.git
          cd nnUNet
          pip install -e .
          ```
3) nnU-Net 需要知道您打算将原始数据、预处理数据和训练模型存放在哪里，因此需要设置若干环境变量。请参考[此处](setting_up_paths.md)的说明。
4) （可选）安装 [hiddenlayer](https://github.com/waleedka/hiddenlayer)。hiddenlayer 使 nnU-Net 能生成其构建的网络拓扑图（参见[模型训练](how_to_use_nnunet.md#model-training)）。安装方法如下：
    ```bash
    pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
    ```

安装 nnU-Net 后，终端中会新增若干命令。这些命令用于运行完整的 nnU-Net 流水线，可在系统中的任意位置执行。所有 nnU-Net 命令都以 `nnUNetv2_` 为前缀，便于识别。

需要注意，这些命令本质上只是执行 Python 脚本。如果您在虚拟环境中安装了 nnU-Net，那么在运行这些命令前需要先激活该环境。您可以查看 [pyproject.toml](../pyproject.toml) 文件中的 project.scripts 了解具体执行了哪些脚本/函数。

所有 nnU-Net 命令都支持 `-h` 选项，用于显示使用说明。
