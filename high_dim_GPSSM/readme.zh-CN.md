# 面向高维动力系统的高效变换 GPSSM

**语言：** [English](./readme.md) | 中文

该目录包含面向高维高斯过程状态空间模型（GPSSMs）的实现，重点是用于非平稳动力系统的高效变换/共享转移模型。

对应的主要参考论文为：

[Efficient Transformed Gaussian Process State-Space Models for Non-Stationary High-Dimensional Dynamical Systems](https://arxiv.org/abs/2503.18309)

除了高效变换模型之外，该子项目还包含基线 EnVI 风格 GPSSM、合成实验、效率对比实验以及若干辅助 demo。

## 包含内容

- **EnVI-GPSSM 基线模型**：采用 EnVI 风格推断、并使用独立转移建模的 GPSSM。
- **高效变换 GPSSM / EGPSSM**：面向高维场景的共享 GP + 学习型变换模型。
- **贝叶斯 / 确定性神经网络转移参数化**：可通过脚本中的 `if_BNN` 和 `if_pureNN` 等开关控制。
- **系统辨识实验**：包含 `actuator`、`ballbeam`、`dryer`、`gasfurnace`、`drive` 等经典真实数据集。
- **非平稳 kink 合成实验**：用于研究非平稳转移动力学。
- **Lorenz-96 效率对比**：比较 GPSSM 与变换 GPSSM 在运行时间和参数量上的差异。
- **演示脚本**：位于 `demo/` 目录下的小型示例与探索性代码。

## 目录结构

- `modules/`
  - `EGPSSM.py`：高效变换/共享 GP GPSSM 的核心实现。
  - `GPSSM_GPyTorch.py`：基线 EnVI 风格 GPSSM 的实现。
  - `Filter.py`、`gpModel.py`、`gpTorch.py`、`inferNet.py`：滤波、GP 和推断相关组件。
  - `realNVP/`：normalizing flow 相关组件。
  - `torchbnn/`：仓库内置的 Bayesian neural network 工具。
- `data/`
  - `real.py`：系统辨识真实数据集加载器。
  - `synthetic.py`：合成数据生成工具，包括 Lorenz-96 和非平稳 kink 示例。
- `demo/`
  - `BNN_demo.py`、`GP.py`、`non_stationary_demo.py`、`sparse_GP.py`：探索性 demo 与辅助脚本。
- 主要实验脚本
  - `_sysID.py`：在系统辨识数据集上运行基线 EnVI-GPSSM。
  - `_sysID_EGP.py`：在系统辨识数据集上运行高效变换 GPSSM。
  - `ns_kink.py`：非平稳 kink 合成实验。
  - `L96_efficiency_comp.py`：基于 Lorenz-96 的计算效率与参数量对比实验。
  - `utils_h.py`：设备、随机种子、评估指标、绘图与模型保存辅助工具。

## 环境与运行说明

### 依赖

该子项目没有单独提供可直接安装的 requirements 文件。仓库根目录中虽然有 `requirements.txt`，但它是表格形式的环境快照，而不是可直接用于 `pip install -r` 的标准格式。

从代码导入情况来看，主要依赖包括 PyTorch、GPyTorch、NumPy、SciPy、Matplotlib 和 tqdm。

### 设备配置

默认情况下，`utils_h.py` 中将设备设置为：

```text
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
```

如果你的机器并不使用 `cuda:3`，建议先修改 `high_dim_GPSSM/utils_h.py` 再运行实验。

另外，`L96_efficiency_comp.py` 会显式使用 CPU 来进行效率对比。

### 数据集路径

`data/real.py` 中的真实数据加载器当前使用了硬编码的绝对路径：

```text
DATA_DIR = '/home/student2/zhidi/gpbl/PycharmProj/EnVI_GPSSM/data/datasets'
```

在你的机器上运行系统辨识相关脚本前，几乎一定需要先把这个路径改成本地实际数据目录。

### 输出目录

大多数脚本会将图像、检查点和结果输出到相对路径 `results/` 下。实际保存位置取决于你从哪个工作目录启动脚本。

## 常用实验

下面的命令默认你从仓库根目录开始执行。

### 1）系统辨识：基线 EnVI-GPSSM

```bash
cd high_dim_GPSSM
python _sysID.py
```

该脚本会在多个基准系统辨识数据集上训练基线 EnVI 风格 GPSSM，并周期性保存模型和预测图像。

### 2）系统辨识：高效变换 GPSSM

```bash
cd high_dim_GPSSM
python _sysID_EGP.py
```

这是高效变换 / 共享 GP 模型的主要入口。模型配置例如 `if_BNN`、`if_pureNN` 等开关直接写在脚本中。

### 3）非平稳 kink 合成实验

```bash
cd high_dim_GPSSM
python ns_kink.py
```

该脚本用于在修改后的非平稳 kink 系统上比较基线模型与高效变换模型。

### 4）Lorenz-96 效率对比

```bash
cd high_dim_GPSSM
python L96_efficiency_comp.py
```

该脚本会在不同状态维度下统计运行时间和参数量，并输出对比图。

## 可复现性说明

- 多个脚本会通过 `utils_h.py` 中的 `reset_seed(0)` 固定随机种子。
- `_sysID_EGP.py` 默认会重复多次实验（`num_repeat = 10`），而 `_sysID.py` 默认只运行一次重复实验。
- 许多实验超参数直接写在脚本内部，而不是通过命令行参数传入。
- 结果会受到本地数据路径、GPU 选择和依赖版本的影响。

## 引用

```bibtex
@article{lin2025efficient,
  title={Efficient Transformed Gaussian Process State-Space Models for Non-Stationary High-Dimensional Dynamical Systems},
  author={Lin, Zhidi and Li, Ying and Yin, Feng and Maro{\~n}as, Juan and Thi{\'e}ry, Alexandre H},
  journal={IEEE Transactions on Signal Processing},
  volume={73},
  pages={5229--5243},
  year={2025}
}
```
