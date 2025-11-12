# nnU-Net 中的残差编码器预设

使用这些预设时，请引用我们最近关于 3D 医学图像分割中严谨验证必要性的论文：

> Isensee, F.<sup>* </sup>, Wald, T.<sup>* </sup>, Ulrich, C.<sup>* </sup>, Baumgartner, M.<sup>* </sup>, Roy, S., Maier-Hein, K.<sup>†</sup>, Jaeger, P.<sup>†</sup> (2024). nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation. arXiv preprint arXiv:2404.09556.

*: 共同第一作者\
<sup>†</sup>: 共同通讯作者

[论文链接](https://arxiv.org/pdf/2404.09556.pdf)


自我们参与 KiTS2019 以来，nnU-Net 就已经支持残差编码器 UNet，但一直鲜为人知。
随着全新的 nnUNetResEncUNet 预设发布，这种情况将会改变 :raised_hands:！尤其是在 KiTS2023 和 AMOS2022 等大型数据集上，它们表现出更优的分割性能！

|                        | BTCV  | ACDC  | LiTS  | BraTS | KiTS  | AMOS  | 显存 |  运行时长 | 架构 | 集成于 nnU |
|------------------------|-------|-------|-------|-------|-------|-------|-------|-----|-------|-----|
|                        | n=30  | n=200 | n=131 | n=1251| n=489 | n=360 |       |     |       |     |
| nnU-Net（原版）[1]      | 83.08 | 91.54 | 80.09 | 91.24 | 86.04 | 88.64 |  7.70 |  9  |  CNN  | 是 |
| nnU-Net ResEnc M       | 83.31 | 91.99 | 80.75 | 91.26 | 86.79 | 88.77 |  9.10 |  12 |  CNN  | 是 |
| nnU-Net ResEnc L       | 83.35 | 91.69 | 81.60 | 91.13 | 88.17 | 89.41 | 22.70 |  35 |  CNN  | 是 |
| nnU-Net ResEnc XL      | 83.28 | 91.48 | 81.19 | 91.18 | 88.67 | 89.68 | 36.60 |  66 |  CNN  | 是 |
| MedNeXt L k3 [2]       | 84.70 | 92.65 | 82.14 | 91.35 | 88.25 | 89.62 | 17.30 |  68 |  CNN  | 是 |
| MedNeXt L k5 [2]       | 85.04 | 92.62 | 82.34 | 91.50 | 87.74 | 89.73 | 18.00 | 233 |  CNN  | 是 |
| STU-Net S [3]          | 82.92 | 91.04 | 78.50 | 90.55 | 84.93 | 88.08 |  5.20 |  10 |  CNN  | 是 |
| STU-Net B [3]          | 83.05 | 91.30 | 79.19 | 90.85 | 86.32 | 88.46 |  8.80 |  15 |  CNN  | 是 |
| STU-Net L [3]          | 83.36 | 91.31 | 80.31 | 91.26 | 85.84 | 89.34 | 26.50 |  51 |  CNN  | 是 |
| SwinUNETR [4]          | 78.89 | 91.29 | 76.50 | 90.68 | 81.27 | 83.81 | 13.10 |  15 |   TF  | 是 |
| SwinUNETRV2 [5]        | 80.85 | 92.01 | 77.85 | 90.74 | 84.14 | 86.24 | 13.40 |  15 |   TF  | 是 |
| nnFormer [6]           | 80.86 | 92.40 | 77.40 | 90.22 | 75.85 | 81.55 |  5.70 |  8  |   TF  | 是 |
| CoTr [7]               | 81.95 | 90.56 | 79.10 | 90.73 | 84.59 | 88.02 |  8.20 |  18 |   TF  | 是 |
| No-Mamba Base          | 83.69 | 91.89 | 80.57 | 91.26 | 85.98 | 89.04 |  12.0 |  24 |  CNN  | 是 |
| U-Mamba Bot [8]        | 83.51 | 91.79 | 80.40 | 91.26 | 86.22 | 89.13 | 12.40 |  24 |  Mam  | 是 |
| U-Mamba Enc [8]        | 82.41 | 91.22 | 80.27 | 90.91 | 86.34 | 88.38 | 24.90 |  47 |  Mam  | 是 |
| A3DS SegResNet [9,11]  | 80.69 | 90.69 | 79.28 | 90.79 | 81.11 | 87.27 | 20.00 |  22 |  CNN  |  否 |
| A3DS DiNTS [10, 11]    | 78.18 | 82.97 | 69.05 | 87.75 | 65.28 | 82.35 | 29.20 |  16 |  CNN  |  否 |
| A3DS SwinUNETR [4, 11] | 76.54 | 82.68 | 68.59 | 89.90 | 52.82 | 85.05 | 34.50 |  9  |   TF  |  否 |

结果来自我们上文提到的论文，所列数值为在各自数据集上进行 5 折交叉验证所得的 Dice 得分。所有模型均从头开始训练。

运行时长（RT）：训练耗时（在单张 Nvidia A100 PCIe 40GB 上测量）\
显存（VRAM）：训练期间使用的 GPU 显存，来自 nvidia-smi 报告\
架构（Arch.）：CNN = 卷积神经网络；TF = Transformer；Mam = Mamba\
集成于 nnU（nnU）：指该架构是否已在 nnU-Net 框架中集成并完成测试（由我们或原作者完成）

## 如何使用新的预设

我们提供三个新预设，分别针对不同的 GPU 显存和算力预算：
- **nnU-Net ResEnc M**：GPU 预算与标准 UNet 配置相近，适用于 9-11GB 显存的 GPU。在 A100 上训练约需 12 小时。
- **nnU-Net ResEnc L**：需要 24GB 显存的 GPU。在 A100 上训练约需 35 小时。
- **nnU-Net ResEnc XL**：需要 40GB 显存的 GPU。在 A100 上训练约需 66 小时。

### **:point_right: 我们推荐将 **nnU-Net ResEnc L** 作为新的 nnU-Net 默认配置！ :point_left:**

新的预设可以通过以下方式使用（从 M/L/XL 中选择其一）：
1. 在执行实验规划与预处理时指定所需配置：  
`nnUNetv2_plan_and_preprocess -d DATASET -pl nnUNetPlannerResEnc(M/L/XL)`。这些规划器与标准的 2d 和 3d_fullres 配置复用同一预处理数据文件夹，因为预处理数据完全一致。仅 `3d_lowres` 会有所不同，并保存到另一个文件夹，以便所有配置共存！如果你只计划运行 3d_fullres/2d，并且已经完成预处理，可以直接运行 `nnUNetv2_plan_experiment -d DATASET -pl nnUNetPlannerResEnc(M/L/XL)` 以避免重复预处理。
2. 在运行 `nnUNetv2_train`、`nnUNetv2_predict` 等命令时指定相应的 plans。所有 nnU-Net 命令的接口保持一致：`-p nnUNetResEncUNet(M/L/XL)Plans`  

新预设的训练结果会保存到专用文件夹，不会覆盖标准 nnU-Net 的训练结果！尽管放心尝试吧！

## 将 ResEnc nnU-Net 扩展到预设之外
这些预设与 `ResEncUNetPlanner` 相比有两个区别：
- 为 `gpu_memory_target_in_gb` 设置了新的默认值，以目标显存消耗为导向；
- 取消了批量大小 0.05 的上限（此前单个批次覆盖的像素不得超过整个数据集的 5%，现在可以任意大）。

预设仅用于简化操作，并提供可供基准评测的标准化配置。你可以轻松调整 GPU 显存目标，以匹配你的 GPU，并扩展到超过 40GB 显存。

以下示例展示了如何在 Dataset003_Liver 上扩展到 80GB 显存：

`nnUNetv2_plan_experiment -d 3 -pl nnUNetPlannerResEncM -gpu_memory_target 80 -overwrite_plans_name nnUNetResEncUNetPlans_80G`

之后按照前面说明，使用 `-p nnUNetResEncUNetPlans_80G` 即可！运行上述示例时会出现警告（“You are running nnUNetPlannerM with a non-standard gpu_memory_target_in_gb”）。在此情境下可以忽略该警告。  
**在调整显存目标时务必通过 `-overwrite_plans_name NEW_PLANS_NAME` 修改 plans 标识，以免覆盖预设的 plans！**

为何不直接使用 `ResEncUNetPlanner`？因为它仍然保留 5% 的上限！

### 扩展到多 GPU
在扩展到多张 GPU 时，不要将多张卡的总显存直接传给 `nnUNetv2_plan_experiment`，否则可能会得到单卡无法处理的 patch 大小。最佳实践是按单张 GPU 的显存预算运行该命令，然后手动编辑 plans 文件以增大 batch size。你可以使用[配置继承](explanation_plans_files.md)。在生成的 plans JSON 文件中的 `configurations` 字典内添加如下条目：

```json
        "3d_fullres_bsXX": {
            "inherits_from": "3d_fullres",
            "batch_size": XX
        },
```
其中 XX 是新的批量大小。如果单卡的 `3d_fullres` 批量大小为 2，而你计划扩展到 8 张 GPU，那么新的批量大小应为 2×8=16！随后可使用 nnU-Net 的多 GPU 配置来训练新设置：

```bash
nnUNetv2_train DATASETID 3d_fullres_bsXX FOLD -p nnUNetResEncUNetPlans_80G -num_gpus 8
```

## 提出新的分割方法？请以正确方式进行基准评测！
在将新分割方法与 nnU-Net 对比时，我们鼓励基准对比残差编码器变体。为了公平比较，请选择在显存和算力需求上与你的方法最接近的变体！


## 参考文献
 [1] Isensee, Fabian, et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature Methods 18.2 (2021): 203-211.\
 [2] Roy, Saikat, et al. "MedNeXt: transformer-driven scaling of convnets for medical image segmentation." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2023.\
 [3] Huang, Ziyan, et al. "STU-Net: Scalable and transferable medical image segmentation models empowered by large-scale supervised pre-training." arXiv preprint arXiv:2304.06716 (2023).\
 [4] Hatamizadeh, Ali, et al. "Swin UNETR: Swin transformers for semantic segmentation of brain tumors in MRI images." International MICCAI Brainlesion Workshop. Cham: Springer International Publishing, 2021.\
 [5] He, Yufan, et al. "SwinUNETR-V2: Stronger Swin transformers with stagewise convolutions for 3D medical image segmentation." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2023.\
 [6] Zhou, Hong-Yu, et al. "nnFormer: Interleaved transformer for volumetric segmentation." arXiv preprint arXiv:2109.03201 (2021).\
 [7] Xie, Yutong, et al. "CoTr: Efficiently bridging CNN and transformer for 3D medical image segmentation." Medical Image Computing and Computer Assisted Intervention–MICCAI 2021: 24th International Conference, Strasbourg, France, September 27–October 1, 2021, Proceedings, Part III 24. Springer International Publishing, 2021.\
 [8] Ma, Jun, Feifei Li, and Bo Wang. "U-Mamba: Enhancing long-range dependency for biomedical image segmentation." arXiv preprint arXiv:2401.04722 (2024).\
 [9] Myronenko, Andriy. "3D MRI brain tumor segmentation using autoencoder regularization." Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries: 4th International Workshop, BrainLes 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 16, 2018, Revised Selected Papers, Part II 4. Springer International Publishing, 2019.\
 [10] He, Yufan, et al. "DiNTS: Differentiable neural network topology search for 3D medical image segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.\
 [11] Auto3DSeg, MONAI 1.3.0, [链接](https://github.com/Project-MONAI/tutorials/tree/ed8854fa19faa49083f48abf25a2c30ab9ac1c6b/auto3dseg)

