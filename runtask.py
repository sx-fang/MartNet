# The core function of this script is run_task.
import ast
import os
import random
import socket
import warnings
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path
from typing import Optional

import numpy as np

import pandas as pd

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import examples
import martnetdf
from martnetdf import EigenFuncValue, DNNtx, PathSampler, train_martnet
from exmeta import HJB, PDE
from examples import EVP
from savresult import plot_summary, save_results, summary_hist

DEFAULT_CONFIG = './default_config.ini'
TASK_PATH = './taskfiles'


def set_seed(config, rank: int = 0):
    seed_str = config.get('Platform', 'seed')
    if seed_str == "None":
        return None
    else:
        seed = ast.literal_eval(seed_str) + rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.xpu.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def clean_mp():
    dist.barrier()
    dist.destroy_process_group()


def kwarg2dict(**kwargs):
    return kwargs


def get_config(path):
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(path, encoding='utf-8')
    return config


def sav_config(config, path):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    with open(f'{path}.ini', 'w') as configfile:
        config.write(configfile)


def set_torchdtype(config):
    dtype = config['Platform']['torch_dtype']
    if dtype == 'float64':
        torch.set_default_dtype(torch.float64)
    elif dtype == 'float32':
        torch.set_default_dtype(torch.float32)
    elif dtype == 'float16':
        torch.set_default_dtype(torch.float16)
    else:
        raise ValueError(f"Unsupported torch.dtype: {dtype}")


def print_gpu():
    for rank in range(torch.cuda.device_count()):
        gpu_i = torch.device(f"cuda:{rank}")
        properties = torch.cuda.get_device_properties(gpu_i)
        print(f"GPU {rank}: {properties.name}")
        print("Memory: ", properties.total_memory)
    for rank in range(torch.xpu.device_count()):
        gpu_i = torch.device(f"xpu:{rank}")
        properties = torch.xpu.get_device_properties(gpu_i)
        print(f"GPU {rank}: {properties.name}")
        print("Memory: ", properties.total_memory)


def parse_device(config):
    dev_conf = config['Platform']['device']
    n_cuda = torch.cuda.device_count()
    n_xpu = torch.xpu.device_count()

    if dev_conf == 'default':
        if n_cuda > 0:
            device = 'cuda'
        elif n_xpu > 0:
            device = 'xpu'
        else:
            device = 'cpu'
    elif dev_conf in ('cuda', 'xpu') and (max(n_cuda, n_xpu) == 0):
        warnings.warn('GPU is unavailable. Use cpu instead.')
        device = 'cpu'
    elif dev_conf == 'cuda' and (n_cuda > 0):
        device = 'cuda'
    elif dev_conf == 'xpu' and (n_xpu > 0):
        device = 'xpu'
    elif dev_conf == 'cpu':
        device = 'cpu'
    else:
        raise ValueError(
            f"Invalid config setting: config[Platform][device] = {dev_conf}")

    if device in ('cuda', 'xpu'):
        print_gpu()
        n_gpu = n_cuda if device == 'cuda' else n_xpu
        ws_str = config['Platform']['world_size']
        if ws_str == 'auto':
            world_size = n_gpu
        else:
            world_size = int(ws_str)
            if world_size > n_gpu:
                world_size = n_gpu
                warnings.warn(
                    f'Available GPUs is less than world_size = {world_size}. \n Reset world_size = {n_gpu}.'
                )
    else:
        world_size = 1
    return device, world_size


def get_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = random.randint(1024, 65535)
    while True:
        try:
            sock.bind(('localhost', port))
            break
        except socket.error:
            port = random.randint(1024, 65535)
    sock.close()
    return port


def init_processgp(rank, world_size):
    try:
        dist.init_process_group(backend="nccl",
                                rank=rank,
                                world_size=world_size)
        if rank == 0:
            print('Using nccl backend')
    except RuntimeError:
        dist.init_process_group(backend="gloo",
                                rank=rank,
                                world_size=world_size)
        if rank == 0:
            print('Nccl is unavailable. Use gloo backend instead.')


def set_master(config):
    mp_str = config.get('Platform', 'master_port')
    if mp_str == 'random':
        master_port = get_free_port()
    else:
        master_port = int(mp_str)
    use_libuv = config.get('Platform', 'use_libuv')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ["USE_LIBUV"] = use_libuv
    print(f'Master Port: {master_port}')


def parse_example(config, use_dist):
    exa_name = config.get('Example', 'name')
    dim_x = config.getint('Example', 'dim_x')
    example = getattr(examples, exa_name)(dim_x, use_dist=use_dist)
    return example


def parse_martnet(config, example: PDE, use_dist=False, rank=0):

    vnn, optim_v, sch_v = parse_v(config,
                                  example,
                                  use_dist=use_dist,
                                  rank=rank)
    rhonn, optim_asc, sch_rho = parse_rho(config,
                                          example,
                                          use_dist=use_dist,
                                          rank=rank)
    if isinstance(example, HJB):
        unn, optim_u, sch_u = parse_u(config,
                                      example,
                                      use_dist=use_dist,
                                      rank=rank)
        nets = (unn, vnn, rhonn)
        optim_desc = (optim_u, optim_v)
        schs = (sch_u, sch_v, sch_rho)
        meth_str = config.get('Example', 'method_soc')
        num_cost_path = config.getint('Example', 'num_cost_path')
        if use_dist is True:
            wd_size = dist.get_world_size()
            num_cost_path = max(1, int(num_cost_path // wd_size))
        log_func = example.produce_logfunc(
            vnn,
            unn,
            num_cost_path=num_cost_path,
        )
    else:
        nets = (vnn, rhonn)
        log_func = example.produce_logfunc(vnn)
        if isinstance(example, EVP):
            meth_str = config.get('Example', 'method_evp')
            optim_lamb, sch_lamb = parse_lamb(config,
                                              vnn.module.eigenval_parameters())
            optim_desc = (optim_v, optim_lamb)
            schs = (sch_v, sch_lamb, sch_rho)
        else:
            meth_str = config.get('Example', 'method_pde')
            optim_desc = optim_v
            schs = (sch_v, sch_rho)

    MartNet = getattr(martnetdf, meth_str)
    martnet = MartNet(example,
                      nets,
                      num_dt=config.getint('Training', 'num_dt'),
                      use_dist=use_dist,
                      rank=rank)
    kwarg_train = {
        "schs": schs,
        "max_iter": config.getint('Training', 'max_iter'),
        "step_desc": config.getint('Training', 'inner_step_descend'),
        "step_asc": config.getint('Training', 'inner_step_ascend'),
        "log_func": log_func,
    }
    return martnet, optim_desc, optim_asc, kwarg_train


def parse_pathsampler(config,
                      example,
                      ctr_func,
                      world_size=1,
                      rank=0) -> callable:
    epochsize = config.getint('Training', 'epochsize')
    epochsize = max(1, int(epochsize // world_size))

    rate_newpath = config.getfloat('Training', 'rate_newpath')
    num_dt = config.getint('Training', 'num_dt')
    path_sampler = PathSampler(example,
                               epochsize,
                               num_dt=num_dt,
                               rank=rank,
                               ctr_func=ctr_func,
                               rate_newpath=rate_newpath)
    return path_sampler.get_pathbat


def nets2dpp(nets, rank):
    return [DDP(nn.to(rank), device_ids=[rank]) for nn in nets]


def parse_u(config, pde: HJB, use_dist=False, rank=0):
    act_u = getattr(nn, config.get('Network', 'act_u'))
    width_u = eval(config.get('Network', 'width_u'))
    num_hidden_u = config.getint('Network', 'num_hidden_u')
    width_u = [pde.dim_x] + [width_u] * num_hidden_u + [pde.dim_u]
    unn = DNNtx(width_u, act_func=act_u)
    if use_dist is True:
        unn = DDP(unn.to(rank), device_ids=[rank])

    optname_u = config.get('Optimizer', 'optimizer_u')
    kwargs_u = ast.literal_eval(config['Optimizer']['kwargs_u'])
    lr0_u = eval(config.get('Optimizer', 'lr0_u'))
    optim_u = getattr(torch.optim, optname_u)(unn.parameters(),
                                              lr=lr0_u,
                                              **kwargs_u)

    decay_gap_u = config.getint('Optimizer', 'decay_stepgap_u')
    decay_rate_u = eval(config.get('Optimizer', 'decay_rate_u'))
    sch_u = torch.optim.lr_scheduler.StepLR(optim_u,
                                            step_size=decay_gap_u,
                                            gamma=decay_rate_u)
    return unn, optim_u, sch_u


def parse_v(config, pde: PDE, use_dist=False, rank=0):
    act_v = getattr(nn, config.get('Network', 'act_v'))
    num_hidden_v = config.getint('Network', 'num_hidden_v')
    width_v = eval(config.get('Network', 'width_v'))
    width_v = [pde.dim_x] + [width_v] * num_hidden_v + [1]
    if isinstance(pde, EVP):
        vnn = EigenFuncValue(width_v,
                             act_func=act_v,
                             init_lamb=pde.lamb_init,
                             fourier_frequency=pde.fourier_frequency)
        vnn = pde.set_vnn_forward(vnn)
    else:
        vnn = DNNtx(width_v, act_func=act_v)
    if use_dist is True:
        vnn = DDP(vnn.to(rank), device_ids=[rank])

    params = vnn.module.eigenfunc_parameters() if isinstance(
        pde, EVP) else vnn.parameters()

    optname_v = config.get('Optimizer', 'optimizer_v')
    kwargs_v = ast.literal_eval(config['Optimizer']['kwargs_v'])
    lr0_v = eval(config.get('Optimizer', 'lr0_v'))
    decay_gap_v = config.getint('Optimizer', 'decay_stepgap_v')
    decay_rate_v = eval(config.get('Optimizer', 'decay_rate_v'))

    optim_v = getattr(torch.optim, optname_v)(params, lr=lr0_v, **kwargs_v)
    sch_v = torch.optim.lr_scheduler.StepLR(optim_v,
                                            step_size=decay_gap_v,
                                            gamma=decay_rate_v)
    return vnn, optim_v, sch_v


def parse_lamb(config, lamb_params: nn.Parameter):
    optname_lamb = config.get('Optimizer', 'optimizer_lamb')
    kwargs_lamb = ast.literal_eval(config['Optimizer']['kwargs_lamb'])
    lr0_lamb = eval(config.get('Optimizer', 'lr0_lamb'))
    optim_lamb = getattr(torch.optim, optname_lamb)(lamb_params,
                                                    lr=lr0_lamb,
                                                    **kwargs_lamb)
    decay_gap_lamb = config.getint('Optimizer', 'decay_stepgap_lamb')
    decay_rate_lamb = eval(config.get('Optimizer', 'decay_rate_lamb'))
    sch_lamb = torch.optim.lr_scheduler.StepLR(optim_lamb,
                                               step_size=decay_gap_lamb,
                                               gamma=decay_rate_lamb)
    return optim_lamb, sch_lamb


def parse_rho(config, pde: PDE, use_dist=False, rank=0):
    act_rho = getattr(nn, config.get('Network', 'act_rho'))
    num_hidden_rho = config.getint('Network', 'num_hidden_rho')
    width_rho = eval(config.get('Network', 'width_rho'))
    width_rho = [pde.dim_x] + [width_rho] * (num_hidden_rho + 1)

    rho_shell_str = config.get('Network', 'rho_shell')
    rho_shell = getattr(torch, rho_shell_str)
    mulscale = config.getboolean('Network', 'mulscale_rho')
    scale_factor = config.getfloat('Network', 'scale_factor')

    shell_test = lambda _x, y: rho_shell(y)
    # if isinstance(pde, EVP):
    #     rhonn = DNNx(width_rho,
    #                  act_func=act_rho,
    #                  shell_func=shell_test,
    #                  multi_scale=mulscale,
    #                  scale_factor=scale_factor)
    # else:
    rhonn = DNNtx(width_rho,
                  act_func=act_rho,
                  shell_func=shell_test,
                  multi_scale=mulscale,
                  scale_factor=scale_factor)
    if use_dist is True:
        rhonn = DDP(rhonn.to(rank), device_ids=[rank])

    optname_rho = config.get('Optimizer', 'optimizer_rho')
    kwargs_rho = ast.literal_eval(config['Optimizer']['kwargs_rho'])
    lr0_rho = eval(config.get('Optimizer', 'lr0_rho'))
    decay_gap_rho = config.getint('Optimizer', 'decay_stepgap_rho')
    decay_rate_rho = eval(config.get('Optimizer', 'decay_rate_rho'))

    optim_rho = getattr(torch.optim, optname_rho)(
        rhonn.parameters(),
        lr=lr0_rho,
        **kwargs_rho,
    )
    sch_rho = torch.optim.lr_scheduler.StepLR(optim_rho,
                                              step_size=decay_gap_rho,
                                              gamma=decay_rate_rho)

    return rhonn, optim_rho, sch_rho


def parse_repeat_time(config):
    rtime = config.getint('Example', 'repeat_time')
    return rtime


def solve(config, use_dist, rank, sav_name=None):
    batsize = config.getint('Training', 'batsize')
    if use_dist is True:
        wd_size = dist.get_world_size()
        batsize = max(1, int(batsize // wd_size))
    else:
        wd_size = 1

    example = parse_example(config, use_dist)
    martnet, optim_desc, optim_asc, kwarg_train = parse_martnet(
        config,
        example,
        use_dist=use_dist,
        rank=rank,
    )
    syspath_as_pilpath = config.getboolean('Training', 'syspath_as_pilpath')
    if syspath_as_pilpath is True:
        ctr_func = martnet.unn if isinstance(example, HJB) else martnet.vnn
    else:
        ctr_func = None
    pathsamp_func = parse_pathsampler(config,
                                      example,
                                      ctr_func,
                                      world_size=wd_size,
                                      rank=rank)
    hist_dict = train_martnet(martnet,
                              pathsamp_func,
                              optim_desc,
                              optim_asc,
                              batsize=batsize,
                              **kwarg_train)

    if sav_name is not None:
        if (rank == 0) or (use_dist is False):
            output_dir = config.get('Platform', 'output_dir')
            sav_path = f"{output_dir}/{sav_name}"
            save_results(martnet, hist_dict, sav_path)
    return example, hist_dict


def task_on_gpu(rank: int,
                gpu_device: str,
                world_size: int,
                config,
                return_dict,
                sav_name: Optional[str] = None):
    assert gpu_device in ('cuda', 'xpu')
    torch.set_default_device(f'{gpu_device}:{rank}')
    use_dist = world_size > 1
    if use_dist is True:
        init_processgp(rank, world_size)
    set_torchdtype(config)
    set_seed(config, rank=rank)

    hdict_list = []
    rep_time = parse_repeat_time(config)
    issav_everytime = config.getboolean('Example',
                                        'save_result_for_every_repeat_time')
    for r in range(rep_time):
        if issav_everytime is True:
            savname_r = f'{sav_name}_r{r}_'
        elif (r == 0) and (sav_name is not None):
            savname_r = f'{sav_name}_r{r}_'
        else:
            savname_r = None
        _, hist_dict = solve(config, use_dist, rank, sav_name=savname_r)
        hdict_list.append(hist_dict)
    if rank == 0:
        summ_df = summary_hist(hdict_list)
        return_dict[rank] = summ_df
        if sav_name is not None:
            output_dir = config.get('Platform', 'output_dir')
            sav_path = f"{output_dir}/{sav_name}_"
            sav_config(config, sav_path)
            summ_df.to_csv(f'{sav_path}summary.csv')
            plot_summary(summ_df, sav_path)
    if use_dist is True:
        clean_mp()


def task_on_cpu(config, sav_name: str):
    set_torchdtype(config)
    set_seed(config)
    torch.set_default_device('cpu')

    hdict_list = []
    rep_time = parse_repeat_time(config)
    issav_everytime = config.getboolean('Example',
                                        'save_result_for_every_repeat_time')
    for r in range(rep_time):
        if issav_everytime is True:
            savname_r = f'{sav_name}_r{r}_'
        elif (r == 0) and (sav_name is not None):
            savname_r = f'{sav_name}_r{r}_'
        else:
            savname_r = None
        _, hist_dict = solve(config, False, None, sav_name=savname_r)
        hdict_list.append(hist_dict)

    hist_df = summary_hist(hdict_list)
    if sav_name is not None:
        output_dir = config.get('Platform', 'output_dir')
        sav_path = f"{output_dir}/{sav_name}_"
        sav_config(config, sav_path)
        hist_df.to_csv(f'{sav_path}summary.csv')
        plot_summary(hist_df, sav_path)
    return hist_df


def run_task(config, sav_name=None) -> pd.DataFrame:
    device, world_size = parse_device(config)
    if device in ('cuda', 'xpu'):
        print(f'Used device: {device} * {world_size}')
        if world_size == 1:
            return_dict = dict()
            task_on_gpu(0,
                        device,
                        world_size,
                        config,
                        return_dict,
                        sav_name=sav_name)
        else:
            set_master(config)
            return_dict = mp.Manager().dict()
            mp.spawn(task_on_gpu,
                     args=(device, world_size, config, return_dict, sav_name),
                     nprocs=world_size,
                     join=True)
        hist_df = return_dict[0]
    else:
        print(f'Used device: {device}')
        hist_df = task_on_cpu(config, sav_name=sav_name)
    return hist_df


def findfiles(base):
    for root, _, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname


def get_taskfiles(path):
    taskfiles = [f for f in findfiles(path) if f.split('.')[-1] == 'ini']
    taskfiles.sort()
    return taskfiles


def main():
    if not os.path.isdir(TASK_PATH):
        os.mkdir(TASK_PATH)
    config_file = get_taskfiles(TASK_PATH)
    if len(config_file) == 0:
        config = get_config(DEFAULT_CONFIG)
        sav_name = Path(DEFAULT_CONFIG).stem
        run_task(config, sav_name=sav_name)
    else:
        for file in config_file:
            print(f'Task starts: {file}')
            config = get_config(file)
            sav_name = Path(file).stem
            run_task(config, sav_name=sav_name)


if __name__ == '__main__':
    main()
