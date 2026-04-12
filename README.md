# Gaussian Process State-Space Models (GPSSMs)

**Language:** English | [中文](README.zh-CN.md)

This repository collects research implementations of several GPSSM families, including EnVI, output-dependent GPSSMs, and transformed/high-dimensional GPSSMs.

## Included Methods

- **EnVI (TSP 2024)**: Ensemble Kalman filtering for non-mean-field and online inference in GPSSMs.  
  Paper: [Ensemble Kalman Filtering Meets Gaussian Process SSM for Non-Mean-Field and Online Inference](https://ieeexplore.ieee.org/document/10643488)
- **Output-Dependent GPSSM (ICASSP 2023)**  
  Paper: [Output-Dependent Gaussian Process State-Space Model](https://ieeexplore.ieee.org/document/10095784)
- **Efficient Multi-Dimensional GPSSMs (ICASSP 2024)**  
  Paper: [Towards Efficient Modeling and Inference in Multi-Dimensional Gaussian Process State-Space Models](https://arxiv.org/pdf/2309.01074.pdf)
- **Efficient Transformed GPSSMs for High-Dimensional Non-Stationary Dynamics (TSP 2025)**  
  Paper: [Efficient Transformed Gaussian Process State-Space Models for Non-Stationary High-Dimensional Dynamical Systems](https://arxiv.org/abs/2503.18309)

The repository also includes reimplementations of benchmark GPSSM baselines such as PRSSM and VCDT.

## Repository Layout

- `main.py`: Main training script for system-identification experiments with ODGPSSM/EGPSSM.
- `models/`: Core model components used by `main.py`.
- `EnVI/`: EnVI and related scripts (`sysID_EnVI.py`, `syn_EnVI.py`, etc.).
- `high_dim_GPSSM/`: High-dimensional transformed GPSSM implementations and demos.
- `Datasets/`: Helper utilities for dataset-related functions.

## Quick Start

### 1) Create an environment

This repo includes dependency snapshots (`requirements.txt` and `EnVI/requirement.txt`) in a `pip list`-style table format. They are useful as references, but may require conversion before direct installation with `pip install -r`.

### 2) Configure dataset paths

Some dataset loaders use hard-coded data directories. Update these paths for your machine before running experiments:

- `models/dataset.py`
- `EnVI/data/real.py`
- `high_dim_GPSSM/data/real.py`

### 3) Run experiments

From the repository root, typical entry points are:

```bash
python main.py
python EnVI/sysID_EnVI.py
python high_dim_GPSSM/_sysID_EGP.py
```

## Reproducibility Notes

- Scripts include fixed-seed settings in places (for example, `reset_seed(0)` in `main.py`).
- Some scripts save checkpoints/results under local `results/` folders.
- For stable reproduction, keep package versions close to the provided snapshots.

## Citation

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
