# 如何设置环境变量

nnU-Net 需要配置一些环境变量，这样它随时都知道原始数据、预处理数据和训练模型存放的位置。不同操作系统的配置方式会有所区别。

这些变量可以永久设置（推荐！），也可以选择在每次调用 nnU-Net 时临时设置。

# Linux 与 MacOS

## 永久设置
找到主目录中的 `.bashrc` 文件，并在文件末尾加入以下内容：

```bash
export nnUNet_raw="/media/fabian/nnUNet_raw"
export nnUNet_preprocessed="/media/fabian/nnUNet_preprocessed"
export nnUNet_results="/media/fabian/nnUNet_results"
```

（记得根据实际情况修改为你自己的路径。）
如果你使用的是其他 shell，例如 zsh，需要找到对应的启动脚本。对 zsh 来说，这个文件是 `.zshrc`。

## 临时设置
每次运行 nnU-Net 前执行以下命令即可：
```bash
export nnUNet_raw="/media/fabian/nnUNet_raw"
export nnUNet_preprocessed="/media/fabian/nnUNet_preprocessed"
export nnUNet_results="/media/fabian/nnUNet_results"
```
（记得根据实际情况修改为你自己的路径。）

重要提示：关闭终端后这些变量会被清除！它们只对当前终端窗口生效，无法在不同终端之间共享！

你也可以在运行 nnU-Net 的命令前直接添加这些变量：

`nnUNet_results="/media/fabian/nnUNet_results" nnUNet_preprocessed="/media/fabian/nnUNet_preprocessed" nnUNetv2_train[...]`

## 验证环境变量是否已设置
可以执行 `echo ${nnUNet_raw}` 等命令来查看环境变量。如果变量未设置，将返回空字符串。

# Windows
参考链接：
- [https://www3.ntu.edu.sg](https://www3.ntu.edu.sg/home/ehchua/programming/howto/Environment_Variables.html#:~:text=To%20set%20(or%20change)%20a,it%20to%20an%20empty%20string.)
- [https://phoenixnap.com](https://phoenixnap.com/kb/windows-set-environment-variable)

## 永久设置
参考 [这里](https://phoenixnap.com/kb/windows-set-environment-variable) 的 `Set Environment Variable in Windows via GUI`。也可以了解如何通过命令提示符使用 setx。

## 临时设置
在运行 nnU-Net 之前执行以下命令：

(PowerShell)
```PowerShell
$Env:nnUNet_raw = "C:/Users/fabian/nnUNet_raw"
$Env:nnUNet_preprocessed = "C:/Users/fabian/nnUNet_preprocessed"
$Env:nnUNet_results = "C:/Users/fabian/nnUNet_results"
```

(Command Prompt)
```Command Prompt
set nnUNet_raw=C:/Users/fabian/nnUNet_raw
set nnUNet_preprocessed=C:/Users/fabian/nnUNet_preprocessed
set nnUNet_results=C:/Users/fabian/fabian/nnUNet_results
```

（记得根据实际情况修改为你自己的路径。）

重要提示：关闭会话后这些变量会被清除！它们只对当前窗口生效，无法在其他会话中使用！

## 验证环境变量是否已设置
在 Windows 中打印变量的方法取决于当前所处的环境：

PowerShell：`echo $Env:[variable_name]`

Command Prompt：`echo %variable_name%`
