# nnU-Netv2 基准测试

你的系统运行是否正常？单个 epoch 的耗时是否比预期更长？你应该期望怎样的 epoch 时间？

别再寻找了，我们已经为你准备好了答案！

## nnU-Netv2 基准测试做什么？

nnU-Net 的基准测试会训练 5 个 epoch。结束后会记录最快的 epoch 时间，以及所使用的 GPU 名称、torch 版本和 cudnn 版本。你可以在对应的 `nnUNet_results` 子目录中找到基准测试的输出（参见下方示例）。别担心，我们还提供了用于收集结果的脚本。或者你也可以直接启动基准测试并查看控制台输出。一切皆有可能，毫无禁忌。

基准测试的实现围绕两个训练器展开：
- `nnUNetTrainerBenchmark_5epochs` 会执行常规训练 5 个 epoch。完成后会写出一个 `.json` 文件，记录最快的 epoch 时间以及使用的 GPU、torch 和 cudnn 版本。适合测试完整流水线的速度（数据加载、增广、GPU 训练）。
- `nnUNetTrainerBenchmark_5epochs_noDataLoading` 与其相同，但不会进行任何数据加载或增广，只会向 GPU 提供伪造数组。适合检查纯粹的 GPU 性能。

## 如何运行 nnU-Netv2 基准测试？
其实很简单，它看起来与常规的 nnU-Net 训练没有区别。

我们为部分 Medical Segmentation Decathlon 数据集提供了参考数据，因为它们易于获取：[点击这里下载](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)。如果你想快速又省事，重点关注任务 2 和 4。下载并解压数据，然后使用 `nnUNetv2_convert_MSD_dataset` 将其转换为 nnU-Net 格式。
为它们运行 `nnUNetv2_plan_and_preprocess`。

随后，对每个数据集运行以下命令（每块 GPU 只运行一个，或依次运行）：

```bash
nnUNetv2_train DATSET_ID 2d 0 -tr nnUNetTrainerBenchmark_5epochs
nnUNetv2_train DATSET_ID 3d_fullres 0 -tr nnUNetTrainerBenchmark_5epochs
nnUNetv2_train DATSET_ID 2d 0 -tr nnUNetTrainerBenchmark_5epochs_noDataLoading
nnUNetv2_train DATSET_ID 3d_fullres 0 -tr nnUNetTrainerBenchmark_5epochs_noDataLoading
```

如果你想手动检查结果，可以在 `nnUNet_results/DATASET_NAME/nnUNetTrainerBenchmark_5epochs__nnUNetPlans__3d_fullres/fold_0/` 文件夹中找到 `benchmark_result.json`（以此为例）。

请注意，如果在不同的 GPU、torch 版本或 cudnn 版本上运行了基准测试，该文件中可能会有多条记录！

如果你想像我们在[结果](#results)部分那样汇总结果，可以查看[汇总代码脚本](../nnunetv2/batch_running/benchmarking/summarize_benchmark_results.py)。你需要在里面修改 torch 版本、cudnn 版本以及要汇总的数据集，然后执行该脚本。所需的具体数值可以在你的 `benchmark_result.json` 文件中找到。

## 结果
我们测试了多种 GPU，并将结果整理在一个[电子表格](https://docs.google.com/spreadsheets/d/12Cvt_gr8XU2qWaE0XJk5jJlxMEESPxyqW0CWbQhTNNY/edit?usp=sharing)中。注意，你可以在底部选择不同的 torch 和 cudnn 版本！表格里可能还有备注，记得阅读！

## 结果解读

结果以 epoch 时间（秒）展示，越低越好（废话）。不同运行之间 epoch 时间会有波动，所以只要在我们报告数值的 5-10% 以内，一切就都挺好。

若不是如此，以下是一些定位问题的思路：

首先比较 `nnUNetTrainerBenchmark_5epochs_noDataLoading` 和 `nnUNetTrainerBenchmark_5epochs` 的表现。如果二者的差值与我们电子表格中的差值差不多，但你两项数据都更差，那问题在 GPU：

- 你确定对比的是正确的 GPU 吗（再次废话）
- 如果是，那可以尝试换一种方式安装 PyTorch。绝不要 `pip install torch`！前往 [PyTorch 安装页面](https://pytorch.org/get-started/locally/)，选择你系统支持的最新 CUDA 版本，然后复制并执行对应命令！无论 pip 还是 conda 都可以。
- 如果仍未解决，建议尝试[从源码编译 PyTorch](https://github.com/pytorch/pytorch#from-source)。虽然更困难，但这是我们 DKFZ 的风格（至少酷的小伙伴们都这么干）。
- 另一个需要考虑的是尝试我们在表格中使用的同一 torch + cudnn 版本。新版本有时会降低性能，偶尔也会有 bug。旧版本往往也会慢不少！
- 最后，一些可能影响 GPU 性能的基础问题：
  - GPU 散热是否充足？用 `nvidia-smi` 检查温度。过热的 GPU 会降频以免自燃。
  - 操作系统是否同时用 GPU 驱动桌面显示？若是，则可能带来性能损失（大概 10%？）。这是正常的。
  - 是否有其他用户在使用 GPU？


如果 `nnUNetTrainerBenchmark_5epochs_noDataLoading`（快）和 `nnUNetTrainerBenchmark_5epochs`（慢）之间差距很大，那么问题可能出在数据加载和增广。提醒一下，nnU-Net 不使用预先增广的图像（离线增广），而是在训练过程中即时生成增广样本（不，不能切换到离线模式）。这要求系统能够快速对图像文件进行部分读取（需要 SSD！），并且 CPU 有足够的性能来执行增广。

请检查以下事项：

- 【CPU 瓶颈】训练期间有多少 CPU 线程在运行？nnU-Net 默认使用 12 个进程进行数据增广。如果你看到这 12 个进程一直满载运行，可以考虑增加用于数据增广的进程数量（前提是 CPU 还有余量！）。增加数量直到你观察到活跃进程数少于设置值（或者直接设到 32 然后忘掉它）。通过设置环境变量 `nnUNet_n_proc_DA` 来实现（Linux：`export nnUNet_n_proc_DA=24`）。关于如何设置可以查看[这里](set_environment_variables.md)。如果 CPU 不支持更多进程（设置超过 CPU 线程数是没有意义的！），那你只能升级系统了！
- 【I/O 瓶颈】如果你没有看到 12 个（或你设置的 `nnUNet_n_proc_DA` 数量）进程在运行，但训练时间仍然很慢，那就打开 `top`（抱歉，Windows 用户，我不知道在 Windows 上怎么做）并查看开始于 `%Cpu (s)` 行的 `wa` 左侧的数值。如果该值 >1.0（随意设定的阈值，本质上是看是否出现异常偏高的 `wa`。正常训练中 `wa` 应接近 0），那说明存储在数据加载上卡住了。确保将 `nnUNet_preprocessed` 指向位于 SSD 上的文件夹。NVMe 优于 SATA。PCIe3 足够用了。推荐顺序读取 3000MB/s。
- 【奇怪问题】有时会遇到一些奇怪问题，尤其是在 batch size 很大、文件很小、patch size 也很小的情况下。在数据加载过程中，nnU-Net 需要为每个训练样本打开并关闭一个文件。想象一下 Dataset004_Hippocampus，在 2D 配置下 batch size 为 366，我们在 A100 上 10 秒内跑完 250 次迭代。那是非常多的文件操作（366 * 250 / 10 = 9150 个文件每秒）。夸张吧。如果这些文件放在某个网络硬盘上（即使是 NVMe），那……大概率凉凉。好消息是 nnU-Net 也为此做好了准备：在 `.bashrc` 中添加 `export nnUNet_keep_files_open=True`，问题即可消除。顺带一提：如果系统不允许你打开足够多的文件，这个设置会带来新问题。可能需要提高允许打开文件的数量。`ulimit -n` 可查看当前限制（仅限 Linux）。不要是 1024 这种数值。把它提高到 65535 对我很管用。关于如何更改这些限制，请查看这个[链接](https://kupczynski.info/posts/ubuntu-18-10-ulimits/)（适用于 Ubuntu 18，其他系统请自行搜索）。

