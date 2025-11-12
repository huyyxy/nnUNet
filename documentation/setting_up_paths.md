# 设置路径

nnU-Net 依赖环境变量来确定原始数据、预处理数据以及训练得到的模型权重存储位置。
要完整使用 nnU-Net 的全部功能，必须设置以下三个环境变量：

1) `nnUNet_raw`：用于存放原始数据集。该文件夹会为每个数据集创建一个子文件夹，命名格式为 DatasetXXX_YYY，其中 XXX 是 3 位标识符（如 001、002、043、999 等），YYY 是唯一的数据集名称。数据集必须符合 nnU-Net 格式，详见 [此处](dataset_format.md)。

    示例目录结构：
    ```
    nnUNet_raw/Dataset001_NAME1
    ├── dataset.json
    ├── imagesTr
    │   ├── ...
    ├── imagesTs
    │   ├── ...
    └── labelsTr
        ├── ...
    nnUNet_raw/Dataset002_NAME2
    ├── dataset.json
    ├── imagesTr
    │   ├── ...
    ├── imagesTs
    │   ├── ...
    └── labelsTr
        ├── ...
    ```

2) `nnUNet_preprocessed`：用于存放预处理后的数据。训练过程中也会从该文件夹读取数据。建议将其放置在访问延迟低、吞吐量高的存储介质上（例如 NVMe SSD，PCIe Gen3 即可）。

3) `nnUNet_results`：指定 nnU-Net 保存模型权重的位置。若下载了预训练模型，也会存放于此。

### 如何设置环境变量

请参阅 [此处](set_environment_variables.md)。