import numpy as np
import torch
import math

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
device='cuda:1'
# device = 'cpu'
dtype = torch.float
torch.set_default_dtype(dtype)
reset_seed(seed=0)

## Constant definitions
pi = torch.tensor(math.pi).to(device)

## Computation constants
quad_points     = 100    # number of quadrature points used in integrations
constant_jitter = None   # if provided, then this jitter value is added always when computing cholesky factors
global_jitter   = None   # if None, then it uses 1e-8 with float 64 and 1-6 with float 32 precission when a cholesky error occurs


def save_best_model(model, optimizer, epoch, result_dir, data_name, jj):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    log_dir = result_dir + f"{data_name}_best_model_Repeat{jj}.pt"
    torch.save(state, log_dir)

def save_models(model, optimizer, epoch, losses, result_dir, data_name, jj, save_model=True):
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
