# Efficient Transformed GPSSMs for High-Dimensional Dynamical Systems

**Language:** English | [中文](./readme.zh-CN.md)

This directory contains the implementation of high-dimensional Gaussian Process State-Space Models (GPSSMs), with an emphasis on efficient transformed/shared-transition variants for non-stationary dynamical systems.

The main reference is:

[Efficient Transformed Gaussian Process State-Space Models for Non-Stationary High-Dimensional Dynamical Systems](https://arxiv.org/abs/2503.18309)

In addition to the efficient transformed model, this subproject also includes baseline EnVI-style GPSSMs, synthetic experiments, efficiency comparisons, and supporting demos.

## What is Included

- **EnVI-GPSSM baseline**: a GPSSM with EnVI-style inference and separate-transition modeling.
- **Efficient transformed GPSSM / EGPSSM**: a higher-dimensional variant using a shared GP together with learned transformations.
- **Bayesian / deterministic neural transition parameterizations**: controlled in scripts through flags such as `if_BNN` and `if_pureNN`.
- **System-identification experiments**: classical real-data benchmarks including `actuator`, `ballbeam`, `dryer`, `gasfurnace`, and `drive`.
- **Synthetic non-stationary kink experiments**: for studying non-stationary transition dynamics.
- **Lorenz-96 efficiency comparison**: runtime and parameter-count comparisons between GPSSM and transformed GPSSM variants.
- **Demos**: small exploratory scripts under `demo/`.

## Directory Structure

- `modules/`
  - `EGPSSM.py`: efficient transformed/shared-GP GPSSM implementation.
  - `GPSSM_GPyTorch.py`: baseline EnVI-style GPSSM implementation.
  - `Filter.py`, `gpModel.py`, `gpTorch.py`, `inferNet.py`: filtering, GP, and inference components.
  - `realNVP/`: normalizing-flow components.
  - `torchbnn/`: bundled Bayesian neural network utilities.
- `data/`
  - `real.py`: loaders for system-identification datasets.
  - `synthetic.py`: synthetic generators, including Lorenz-96 and non-stationary kink-type examples.
- `demo/`
  - `BNN_demo.py`, `GP.py`, `non_stationary_demo.py`, `sparse_GP.py`: exploratory demos and utilities.
- Main experiment scripts
  - `_sysID.py`: baseline EnVI-GPSSM on system-identification datasets.
  - `_sysID_EGP.py`: efficient transformed GPSSM on system-identification datasets.
  - `ns_kink.py`: synthetic non-stationary kink experiment.
  - `L96_efficiency_comp.py`: computational-efficiency and parameter-count comparison on Lorenz-96 data.
  - `utils_h.py`: device, seed, metrics, plotting, and checkpoint helpers.

## Setup Notes

### Dependencies

This subproject does not include its own standalone installable requirements file. The repository root contains `requirements.txt`, but it is a package snapshot in table format rather than a ready-to-install `pip install -r` file.

From the code, the main runtime dependencies include PyTorch, GPyTorch, NumPy, SciPy, Matplotlib, and tqdm.

### Device configuration

By default, `utils_h.py` sets the device as:

```text
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
```

If your machine does not use GPU index `3`, you should edit `high_dim_GPSSM/utils_h.py` before running experiments.

`L96_efficiency_comp.py` explicitly forces CPU execution for its comparison script.

### Dataset paths

The real-data loader in `data/real.py` currently uses a hard-coded absolute dataset path:

```text
DATA_DIR = '/home/student2/zhidi/gpbl/PycharmProj/EnVI_GPSSM/data/datasets'
```

You will almost certainly need to update this path on your own machine before running the system-identification scripts.

### Output paths

Most scripts save figures, checkpoints, and summary outputs under relative `results/` directories. The final location depends on the working directory from which you launch the script.

## Typical Experiments

The following commands assume you start from the repository root.

### 1) Baseline EnVI-GPSSM on system-identification datasets

```bash
cd high_dim_GPSSM
python _sysID.py
```

This script trains a baseline EnVI-style GPSSM on several benchmark datasets and periodically saves checkpoints and prediction plots.

### 2) Efficient transformed GPSSM on system-identification datasets

```bash
cd high_dim_GPSSM
python _sysID_EGP.py
```

This is the main entry point for the efficient transformed/shared-GP model. The script exposes modeling switches such as `if_BNN` and `if_pureNN` directly in the source.

### 3) Non-stationary synthetic kink experiment

```bash
cd high_dim_GPSSM
python ns_kink.py
```

This script compares baseline and efficient transformed models on a modified non-stationary kink system.

### 4) Lorenz-96 efficiency comparison

```bash
cd high_dim_GPSSM
python L96_efficiency_comp.py
```

This script measures runtime and parameter count across different state dimensions and saves a comparison plot.

## Reproducibility Notes

- Several scripts call `reset_seed(0)` through `utils_h.py`.
- `_sysID_EGP.py` repeats experiments multiple times by default (`num_repeat = 10`), while `_sysID.py` defaults to a single repeat.
- Many experiment settings are configured directly in the scripts rather than exposed through command-line arguments.
- Results depend on the local dataset path, selected GPU, and package versions.

## Citation

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

