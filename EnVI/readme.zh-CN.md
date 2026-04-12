# 面向高斯过程状态空间模型的 EnVI

**语言：** [English](readme.md) | 中文

该目录包含用于高斯过程状态空间模型（GPSSMs）的 EnVI 方法族实现，核心对应论文为：

[Ensemble Kalman Filtering Meets Gaussian Process SSM for Non-Mean-Field and Online Inference](https://ieeexplore.ieee.org/document/10643488)

除了批量版 EnVI 之外，本子项目还包含在线变体以及用于合成数据、系统辨识和预测任务的相关实验脚本。

## 包含内容

- **EnVI**：基于集合卡尔曼滤波的 GPSSM 批量变分推断方法。
- **Online EnVI / OEnVI**：面向流式观测的顺序/在线更新版本。
- **EnVI-TGP 变体**：用于相关系统辨识实验的 transformed-GP 版本。
- **合成实验**：基于 kink function 的状态空间模拟实验。
- **系统辨识实验**：包含 `actuator`、`ballbeam`、`dryer`、`gasfurnace`、`drive` 等经典数据集。
- **NASCAR 示例**：基于 RSLDS 生成轨迹的在线滤波与预测实验。

## 目录结构

- `models/`
  - `EnVI.py`：EnVI 相关模型的核心实现，包括 `GPSSMs`、`EnVI`、`OEnVI` 和 `OnlineEnVI`。
  - `EnVI_TGP.py`：供 `sysID_TGPSSM.py` 使用的 transformed-GP 变体。
  - `GPModels.py`、`InferNet.py`、`evaluation.py` 等：GP 组件、推断网络与评估辅助模块。
- `data/`
  - `real.py`：系统辨识数据集加载器。
  - `synthetic.py`：合成状态空间数据的生成与绘图工具。
- `utils/`
  - `settings.py`：全局随机种子、设备、数据类型及结果保存工具。
  - `plotResult.py`：供合成/在线实验使用的绘图辅助函数。
- 实验脚本
  - `sysID_EnVI.py`：在系统辨识数据集上运行批量 EnVI。
  - `sysID_TGPSSM.py`：在系统辨识任务上运行 transformed-GP 变体。
  - `syn_EnVI.py`：在合成 kink function 数据上运行批量 EnVI。
  - `syn_OEnVI.py`：在合成数据上运行在线 EnVI/OEnVI。
  - `syn_OEnVI_new.py`：更新版在线 EnVI 实验脚本。
  - `nascar_OEnVI.py`：NASCAR/RSLDS 在线实验。
  - `demo_EnVI.py`：源码中已标记为 outdated 的旧版演示脚本。

## 环境与运行说明

### 依赖

该目录中提供了 `requirement.txt`，但它是 `pip list` 风格的环境快照，而不是可以直接用于安装的标准 requirements 文件。

更稳妥的做法是先准备包含脚本核心依赖的 Python 环境，例如 PyTorch、GPyTorch、NumPy、SciPy、Matplotlib 和 tqdm，再参考 `requirement.txt` 调整具体版本。

### 设备配置

默认情况下，`utils/settings.py` 中设置为：

- `device = 'cuda:0'`
- `dtype = torch.float`

如果你希望在 CPU 或其他 GPU 上运行，请先修改 `EnVI/utils/settings.py`。

### 数据集路径

`data/real.py` 中的真实数据加载器默认使用：

```python
DATA_DIR = 'data/datasets'
```

该路径是相对于当前工作目录的相对路径。实践中，最省事的方式是先进入 `EnVI/` 目录再运行脚本；或者直接把 `DATA_DIR` 改成你本机的数据集路径。

### 输出目录

大多数脚本会将检查点、图像和结果保存到相对路径 `results/` 下。实际保存位置取决于你从哪个目录启动脚本。

## 常用实验

下面的命令默认你从仓库根目录开始执行。

### 1）系统辨识：批量 EnVI

```bash
cd EnVI
python sysID_EnVI.py
```

该脚本会在多个基准系统辨识数据集上训练 EnVI，并周期性保存模型检查点和预测图像。

### 2）合成 kink function：批量 EnVI

```bash
cd EnVI
python syn_EnVI.py
```

该脚本会生成合成轨迹、训练批量 EnVI 模型，并保存拟合曲线与可视化结果。

### 3）合成数据：在线 EnVI

```bash
cd EnVI
python syn_OEnVI_new.py
```

你也可以参考 `syn_OEnVI.py`，其中包含一个非常接近的在线实验配置。

### 4）系统辨识：transformed-GP 变体

```bash
cd EnVI
python sysID_TGPSSM.py
```

### 5）NASCAR 在线滤波/预测示例

```bash
cd EnVI
python nascar_OEnVI.py
```

运行 `nascar_OEnVI.py` 之前，请先确认以下外部文件已存在于对应相对路径下：

- `svmc-main/notebooks/rslds_nascar.npy`

## 可复现性说明

- 多个脚本会通过 `utils/settings.py` 中的 `reset_seed(...)` 显式固定随机种子。
- `sysID_EnVI.py` 和 `sysID_TGPSSM.py` 会进行多次重复实验，并在检测到已有检查点时尝试继续加载。
- 默认配置优先使用 GPU；如果不修改 `utils/settings.py`，脚本会尝试使用 `cuda:0`。
- 实验结果会受到工作目录、数据集本地路径和依赖版本的影响。

## 引用

```bibtex
@article{lin2024EnVI,
  title={Ensemble Kalman Filtering Meets Gaussian Process SSM for Non-Mean-Field and Online Inference},
  author={Lin, Zhidi and Sun, Yiyong and Yin, Feng and Thiery, Alexandre},
  journal={IEEE Transactions on Signal Processing},
  volume={72},
  pages={4286--4301},
  year={2024}
}
```


