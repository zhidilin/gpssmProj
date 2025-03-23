import numpy as np
import torch
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

def reset_seed(seed: int) -> None:
    # 生成随机数，以便固定后续随机数，方便复现代码
    np.random.seed(seed)
    # 没有使用GPU的时候设置的固定生成的随机数
    np.random.seed(seed)
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.manual_seed(seed)
    # torch.cuda.manual_seed()为当前GPU设置随机种子
    torch.cuda.manual_seed(seed)



## Config Variables
# torch_version = '1.8.1'
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
dtype = torch.float
torch.set_default_dtype(dtype)
reset_seed(seed=0)

## Constant definitions
pi = torch.tensor(math.pi).to(device)

## Computation constants
quad_points     = 100    # number of quadrature points used in integrations
constant_jitter = None   # if provided, then this jitter value is added always when computing cholesky factors
global_jitter   = None   # if None, then it uses 1e-8 with float 64 and 1-6 with float 32 precission when a cholesky error occurs

# define measure metric
def quantile_loss(target, pred, q):
    '''
    Calculate the quantile loss for quantile forecasts.
    :param target:
    :param pred:
    :param q:
    :return:
    '''
    return 2 * np.sum(np.abs((pred - target) * ((target <= pred) * 1.0 - q)), axis=1)


def calc_quantile_CRPS(states, assimilated_states):
    '''
    Calculate the Continuous Ranked Probability Score (CRPS) for quantile forecasts.
    A lower CRPS value indicates a better alignment between the estimated posterior and observed distributions,
     indicating a more accurate estimate.
    :param states:
    :param assimilated_states:
    :return:
    '''
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = np.sum(np.abs(states), axis=1)   # (steps, )
    CRPS = np.zeros(states.shape[0])
    for i in range(len(quantiles)):
        pred = np.quantile(assimilated_states, quantiles[i], axis=1)  # (steps, dim)
        q_loss = quantile_loss(states, pred, quantiles[i])            # (steps, )
        CRPS += q_loss / denom
    return CRPS / len(quantiles)

def measure_calculate(states, ensembles):
    '''
    Calculate the RMSE, coverage probability, spread and CRPS for the state estimation.
    :param states:     (steps, dim)
    :param ensembles: (steps, ensemble_size, dim)
    :return:    RMSE, coverage probability, spread and CRPS

    '''
    # post-processing: compute the state estimation RMSE, coverage probability, spread and CRPS
    _mean = torch.mean(ensembles, dim=1)  # (steps, dim)
    average_rmse = torch.sqrt(torch.mean((states - _mean) ** 2, dim=1))  # (steps, )
    variances = torch.var(ensembles, dim=1)  # (steps, dim)
    average_spread = torch.mean(torch.sqrt(variances), dim=1)  # (steps, )
    low_quantile = torch.quantile(ensembles, q=0.1, dim=1)  # (steps, dim)
    high_quantile = torch.quantile(ensembles, q=0.9, dim=1)  # (steps, dim)
    average_coverage = torch.sum((states > low_quantile) & (states < high_quantile), dim=1) / states.shape[1]
    average_crps = calc_quantile_CRPS(states.detach().cpu().numpy(), ensembles.detach().cpu().numpy())
    print("\nPerformance Metrics: ----------------------")
    print(f"RMSE: {average_rmse.mean().item():.4f}")
    print(f"Coverage Probability: {average_coverage.mean().item():.4f}")
    print(f"Spread: {average_spread.mean().item():.4f}")
    print(f"CRPS: {average_crps.mean().item():.4f}")
    return average_rmse.mean().item(), average_coverage.mean().item(), average_spread.mean().item(), average_crps.mean().item()


def plot_lorenz_trajectory_all(states,
                               assimilated_states_1,
                               assimilated_states_2,
                               assimilated_states_3,
                               observations,  # 新增参数
                               steps, save_name='DEnF_Lorenz96'):
    mpl.rcdefaults()
    mpl.rc("mathtext", fontset="cm")
    mpl.rc("font", family="serif", serif="DejaVu Serif")
    mpl.rc("figure", dpi=600, titlesize=9)
    mpl.rc("figure.subplot", wspace=0.2, hspace=0.6)
    mpl.rc("axes", grid=False, labelsize=9, labelpad=0.5)
    mpl.rc("axes.spines", top=False, right=False)
    mpl.rc("xtick", labelsize=6, direction="out")
    mpl.rc("ytick", labelsize=6, direction="out")
    mpl.rc("xtick.major", pad=2)
    mpl.rc("ytick.major", pad=2)
    mpl.rc("grid", linestyle=":", alpha=0.8)
    mpl.rc("lines", linewidth=1, markersize=5, markerfacecolor="none", markeredgecolor="auto", markeredgewidth=0.5)
    mpl.rc("scatter", marker='o')
    mpl.rc("legend", fontsize=9)

    dt = 6 / (states.shape[0])
    states = states.cpu()  # (steps, dim)
    observations = observations.cpu()  # 新增：确保 observations 在 CPU 上
    assimilated_states = assimilated_states_1.cpu()  # (steps, nsample, dim)
    mean_estimation = torch.mean(assimilated_states, dim=1)         # (steps, dim)
    mean_estimation_2 = torch.mean(assimilated_states_2.cpu(), dim=1)  # (steps, dim)
    mean_estimation_3 = torch.mean(assimilated_states_3.cpu(), dim=1)  # (steps, dim)
    _, dim = mean_estimation.shape

    # **从 100 维中均匀采样 9 维**
    sampled_indices = np.linspace(0, dim - 1, 9, dtype=int)  # 生成 9 个均匀间隔的索引

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(7, 4))
    t = np.arange(steps) * dt

    for idx, ax in enumerate(axes.flat):  # 直接遍历 3x3 轴
        dim_idx = sampled_indices[idx]  # 取出对应的维度索引
        ax.scatter(t[::2], states[:, dim_idx][::2],
                   label='Truth',
                   color='C0',
                   marker='o',
                   facecolors='none',
                   s=4,
                   linewidths=1.)  # 真实值

        # **绘制观察值**
        ax.scatter(t[::5], observations[:, dim_idx][::5],
                   label='Observations',
                   color='black',  # 颜色设置为浅灰色
                   marker='x',
                   s=6,
                   alpha=0.5)  # 调整点大小

        ax.plot(t[::2], mean_estimation[:, dim_idx][::2], label='AD-EnKF', color='C1', markevery=1)
        ax.plot(t[::2], mean_estimation_2[:, dim_idx][::2], label='ETGPSSM', color='C2', markevery=1)
        ax.plot(t[::2], mean_estimation_3[:, dim_idx][::2], label='EnKF', color='C3', markevery=1)

        ax.set_title(f"$x_{{{dim_idx + 1}}}$")  # 标题改为选取的维度

    axes[0][1].legend(bbox_to_anchor=(-1.3, 1.3), loc='lower left', ncol=5)
    plt.text(0.08, 0.5, 'State', transform=plt.gcf().transFigure, fontsize=9, rotation='vertical')
    plt.text(0.43, 0.03, 'Time step', transform=plt.gcf().transFigure, fontsize=9)
    plt.savefig(f'{save_name}.pdf', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    return fig



# some function for saving results
def save_best_model(model, optimizer, epoch, result_dir, data_name, jj):
    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch}
    log_dir = result_dir + f"{data_name}_best_model_Repeat{jj}.pt"
    torch.save(state, log_dir)

def save_models(model, optimizer, epoch, losses, result_dir, data_name, jj, save_model=True):
    '''

    Parameters
    ----------
    model
    optimizer
    epoch
    losses
    result_dir  :           result saving path
    data_name   :           data name
    jj          :           number of experiment repetition
    save_model  :           indication if to saving model

    Returns
    -------

    '''
    state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch,
             'losses': losses}
    if save_model:
        log_dir = result_dir + f"{data_name}_epoch{epoch}_Repeat{jj}.pt"
        torch.save(state, log_dir)

def save_results(RMSE, log_ll, result_dir, jj):
    state = {'RMSE': RMSE,
            'log_ll': log_ll
            }
    log_dir = result_dir + f"results_repeat{jj}.pt"
    torch.save(state, log_dir)
