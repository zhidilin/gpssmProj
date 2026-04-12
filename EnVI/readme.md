# EnVI for Gaussian Process State-Space Models

**Language:** English | [中文](./readme.zh-CN.md)

This directory contains the implementation of the EnVI family of methods for Gaussian Process State-Space Models (GPSSMs), centered on the paper:

[Ensemble Kalman Filtering Meets Gaussian Process SSM for Non-Mean-Field and Online Inference](https://ieeexplore.ieee.org/document/10643488)

In addition to the batch EnVI model, this subproject also includes online variants and related experiment scripts used for synthetic, system-identification, and forecasting tasks.

## What is Included

- **EnVI**: batch variational inference for GPSSMs using ensemble Kalman filtering.
- **Online EnVI / OEnVI**: sequential/online updates for streaming observations.
- **EnVI-TGP variant**: a transformed-GP version used in related system-identification experiments.
- **Synthetic experiments**: kink-function style state-space simulations.
- **System identification experiments**: classical benchmark datasets such as `actuator`, `ballbeam`, `dryer`, `gasfurnace`, and `drive`.
- **NASCAR example**: an online filtering and forecasting experiment based on an RSLDS-generated trajectory.

## Directory Structure

- `models/`
  - `EnVI.py`: core implementations of EnVI-related models, including `GPSSMs`, `EnVI`, `OEnVI`, and `OnlineEnVI`.
  - `EnVI_TGP.py`: transformed-GP variant used by `sysID_TGPSSM.py`.
  - `GPModels.py`, `InferNet.py`, `evaluation.py`, etc.: GP building blocks, inference networks, and evaluation helpers.
- `data/`
  - `real.py`: loaders for system-identification datasets.
  - `synthetic.py`: generators and plotting utilities for synthetic state-space data.
- `utils/`
  - `settings.py`: global seed, device, dtype, and result-saving utilities.
  - `plotResult.py`: plotting helpers used by synthetic/online experiments.
- Experiment scripts
  - `sysID_EnVI.py`: batch EnVI on system-identification datasets.
  - `sysID_TGPSSM.py`: transformed-GP variant on system-identification datasets.
  - `syn_EnVI.py`: batch EnVI on synthetic kink-function data.
  - `syn_OEnVI.py`: online EnVI/OEnVI on synthetic data.
  - `syn_OEnVI_new.py`: updated online EnVI experiment script.
  - `nascar_OEnVI.py`: online EnVI on the NASCAR/RSLDS example.
  - `demo_EnVI.py`: legacy demo script marked as outdated in the source file.

## Setup Notes

### Dependencies

This folder includes `requirement.txt`, but it is a package snapshot in `pip list`-style table format rather than a ready-to-install requirements file.

A practical starting point is to prepare a Python environment with the core packages imported by the scripts, such as PyTorch, GPyTorch, NumPy, SciPy, Matplotlib, and tqdm, then adjust versions as needed based on `requirement.txt`.

### Device configuration

By default, `utils/settings.py` sets:

- `device = 'cuda:0'`
- `dtype = torch.float`

If you want to run on CPU or a different GPU, edit `EnVI/utils/settings.py` before launching experiments.

### Dataset paths

The real-data loader in `data/real.py` uses:

```python
DATA_DIR = 'data/datasets'
```

This path is relative to the current working directory. In practice, the easiest option is to run scripts from inside `EnVI/`, or to modify `DATA_DIR` to match your local dataset location.

### Output paths

Most scripts save checkpoints, plots, and summary results under a relative `results/` directory. The exact output location depends on the directory from which you launch the script.

## Typical Experiments

The following commands assume you start from the repository root.

### 1) Batch EnVI on system-identification datasets

```bash
cd EnVI
python sysID_EnVI.py
```

This script trains EnVI on several benchmark datasets and periodically saves checkpoints and prediction plots.

### 2) Batch EnVI on synthetic kink-function data

```bash
cd EnVI
python syn_EnVI.py
```

This script generates synthetic trajectories, trains a batch EnVI model, and saves fitting curves and visualizations.

### 3) Online EnVI on synthetic data

```bash
cd EnVI
python syn_OEnVI_new.py
```

You can also inspect `syn_OEnVI.py`, which contains a closely related online experiment setup.

### 4) Transformed-GP system-identification variant

```bash
cd EnVI
python sysID_TGPSSM.py
```

### 5) NASCAR online filtering/forecasting example

```bash
cd EnVI
python nascar_OEnVI.py
```

Before running `nascar_OEnVI.py`, make sure the external file below exists at the expected relative path:

- `svmc-main/notebooks/rslds_nascar.npy`

## Reproducibility Notes

- Several scripts explicitly fix random seeds through `reset_seed(...)` in `utils/settings.py`.
- `sysID_EnVI.py` and `sysID_TGPSSM.py` repeat experiments multiple times and reuse checkpoints if matching files already exist.
- The default configuration uses GPU execution unless you manually switch `device` in `utils/settings.py`.
- Results are sensitive to working directory, local dataset layout, and dependency versions.

## Citation

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