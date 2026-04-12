# 高斯过程状态空间模型（GPSSMs）

**语言:** [English](README.md) | 中文

本仓库汇集了多个 GPSSM 方法族的研究实现，包括 EnVI、输出相关 GPSSM，以及面向高维场景的变换型 GPSSM。

## 包含的方法

- **EnVI（TSP 2024）**：将集合卡尔曼滤波用于 GPSSM 的非均值场（non-mean-field）与在线推断。  
  论文：[Ensemble Kalman Filtering Meets Gaussian Process SSM for Non-Mean-Field and Online Inference](https://ieeexplore.ieee.org/document/10643488)
- **输出相关 GPSSM（ICASSP 2023）**  
  论文：[Output-Dependent Gaussian Process State-Space Model](https://ieeexplore.ieee.org/document/10095784)
- **高效多维 GPSSM（ICASSP 2024）**  
  论文：[Towards Efficient Modeling and Inference in Multi-Dimensional Gaussian Process State-Space Models](https://arxiv.org/pdf/2309.01074.pdf)
- **面向高维非平稳动力系统的高效变换 GPSSM（TSP 2025）**  
  论文：[Efficient Transformed Gaussian Process State-Space Models for Non-Stationary High-Dimensional Dynamical Systems](https://arxiv.org/abs/2503.18309)

本仓库也包含了 PRSSM 和 VCDT 等基准 GPSSM 方法的复现实现。

## 仓库结构

- `main.py`：使用 ODGPSSM/EGPSSM 进行系统辨识实验的主训练脚本。
- `models/`：`main.py` 使用的核心模型组件。
- `EnVI/`：EnVI 及相关脚本（如 `sysID_EnVI.py`、`syn_EnVI.py` 等）。
- `high_dim_GPSSM/`：高维变换 GPSSM 的实现与示例。
- `Datasets/`：与数据集相关的辅助工具。

## 快速开始

### 1）创建环境

本仓库包含依赖快照（`requirements.txt` 和 `EnVI/requirement.txt`），其格式为 `pip list` 风格表格。它们可作为版本参考，但通常需要先转换格式后才能直接用于 `pip install -r`。

### 2）配置数据集路径

部分数据加载器使用了硬编码的数据目录。运行实验前，请根据本机环境修改以下文件中的路径：

- `models/dataset.py`
- `EnVI/data/real.py`
- `high_dim_GPSSM/data/real.py`

### 3）运行实验

在仓库根目录下，常用入口如下：

```bash
python main.py
python EnVI/sysID_EnVI.py
python high_dim_GPSSM/_sysID_EGP.py
```

## 可复现性说明

- 部分脚本包含固定随机种子设置（例如 `main.py` 中的 `reset_seed(0)`）。
- 部分脚本会在本地 `results/` 目录保存检查点与实验结果。
- 如需稳定复现实验，建议尽量贴近仓库中提供的依赖版本快照。

## 引用

```bibtex
@inproceedings{lin2023output,
  title={Output-Dependent Gaussian Process State-Space Model},
  author={Lin, Zhidi and Cheng, Lei and Yin, Feng and Xu, Lexi and Cui, Shuguang},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}

@inproceedings{lin2024towards,
  title={Towards Efficient Modeling and Inference in Multi-Dimensional Gaussian Process State-Space Models},
  author={Lin, Zhidi and Maro\~{n}as, Juan and Li, Ying and Yin, Feng and Theodoridis, Sergios},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={12881--12885},
  year={2024},
  organization={IEEE}
}

@article{lin2023EnVI,
  title={Ensemble Kalman Filtering Meets Gaussian Process SSM for Non-Mean-Field and Online Inference},
  author={Lin, Zhidi and Sun, Yiyong and Yin, Feng and Thiery, Alexandre},
  journal={IEEE Transactions on Signal Processing},
  volume={72},
  pages={4286--4301},
  year={2024}
}

@article{lin2025efficient,
  title={Efficient Transformed Gaussian Process State-Space Models for Non-Stationary High-Dimensional Dynamical Systems},
  author={Lin, Zhidi and Li, Ying and Yin, Feng and Maro{\~n}as, Juan and Thi{\'e}ry, Alexandre H},
  journal={IEEE Transactions on Signal Processing},
  volume={73},
  pages={5229--5243},
  year={2025}
}
```

