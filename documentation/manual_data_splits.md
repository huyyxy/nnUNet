# 如何在 nnU-Net 中生成自定义划分

有时候，nnU-Net 默认的 5 折交叉验证划分并不适合你的项目。也许你想改为 3 折交叉验证？或者你的训练病例不能随机划分，需要按特定策略分层？别担心，nnU-Net 也能满足这些需求（它真的无所不能 <3）。

nnU-Net 使用的划分由 nnUNetTrainer 的 `do_split` 函数生成。该函数会先查找是否存在已经保存的划分文件；如果没有，就会自动创建一个。因此，如果你想自定义划分，最好的方式就是手动创建一个相应的划分文件，让它能够被识别和使用。

划分文件位于 `nnUNet_preprocessed/DATASETXXX_NAME` 文件夹中。因此，最佳实践是先运行 `nnUNetv2_plan_and_preproccess`，以便填充这个文件夹。

划分以 .json 文件形式保存，本质上是一个 Python 列表。列表长度对应划分数量（默认情况下为 5）。列表的每个元素都是一个包含 `train` 和 `val` 键的字典，各自的值又是包含训练/验证病例标识的列表。下面用 Dataset002 的文件作为例子：

```commandline
In [1]: from batchgenerators.utilities.file_and_folder_operations import load_json

In [2]: splits = load_json('splits_final.json')

In [3]: len(splits)
Out[3]: 5

In [4]: splits[0].keys()
Out[4]: dict_keys(['train', 'val'])

In [5]: len(splits[0]['train'])
Out[5]: 16

In [6]: len(splits[0]['val'])
Out[6]: 4

In [7]: print(splits[0])
{'train': ['la_003', 'la_004', 'la_005', 'la_009', 'la_010', 'la_011', 'la_014', 'la_017', 'la_018', 'la_019', 'la_020', 'la_022', 'la_023', 'la_026', 'la_029', 'la_030'],
'val': ['la_007', 'la_016', 'la_021', 'la_024']}
```

如果你仍然不确定划分文件应该长什么样，不妨从 [Medical Decathlon](http://medicaldecathlon.com/) 下载一个参考数据集，启动一次训练（以生成划分），然后用你喜欢的文本编辑器打开该 .json 文件进行查看。

要生成自定义划分，你只需要按照上述结构构建数据，并将其保存为 `nnUNet_preprocessed/DATASETXXX_NAME` 文件夹下的 `splits_final.json`。之后即可像往常一样运行 `nnUNetv2_train` 等命令。