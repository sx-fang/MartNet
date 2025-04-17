# Produce task files, which will be used by runtask.py

from itertools import product
from runtask import (get_config, sav_config, TASK_PATH, DEFAULT_CONFIG)


def maxit_default(dim_x):
    if dim_x < 1000:
        maxit = 2000
    elif dim_x < 5000:
        maxit = 4000
    else:
        maxit = 6000
    return maxit


def maxit_long(dim_x):
    if dim_x < 1000:
        maxit = 3000
    elif dim_x < 5000:
        maxit = 6000
    else:
        maxit = 9000
    return maxit


def batsz(dim_x):
    if dim_x < 1000:
        bsz = 256
    elif dim_x < 5000:
        bsz = 128
    else:
        bsz = 64
    return bsz


def default_lrfunc(dim_x):
    if dim_x >= 1000:
        expo = 0.8
    else:
        expo = 0.5
    lr0_u = 3 * 1e-3 / dim_x**expo
    lr0_v = 3 * 1e-3 / dim_x**expo
    lr0_rho = 3 * 1e-2 / dim_x**expo
    return lr0_u, lr0_v, lr0_rho


def ev_lrfunc(dim_x):
    if dim_x >= 1000:
        expo = 0.8
    else:
        expo = 0.5
    lr0_u = 3 * 1e-4 / dim_x**expo
    lr0_v = 3 * 1e-4 / dim_x**expo
    lr0_rho = 3 * 1e-3 / dim_x**expo
    return lr0_u, lr0_v, lr0_rho


def var_width(dim_x):
    if dim_x <= 100:
        wid = 5 * dim_x + 10
    elif dim_x <= 1000:
        wid = 2 * dim_x + 10
    else:
        wid = dim_x + 10
    return wid


def make_task(
    dimx_list,
    example_list,
    numdt_list,
    num_hidd=4,
    start_idx=0,
    lr_func=default_lrfunc,
    maxit_func=None,
    width_func=var_width,
    repeat_time=1,
):
    idx = start_idx
    tot_idx = start_idx - 1 + len(dimx_list) * len(example_list)
    length = len(str(tot_idx))

    if maxit_func is None:
        maxit_func = maxit_default

    for dimx, exmaple, num_dt in product(dimx_list, example_list, numdt_list):
        bsz = batsz(dimx)
        wid = width_func(dimx)
        config = get_config(DEFAULT_CONFIG)
        config.set('Example', 'dim_x', str(dimx))
        config.set('Network', 'width_u', str(wid))
        config.set('Network', 'width_v', str(wid))
        config.set('Network', 'num_hidden_u', str(num_hidd))
        config.set('Network', 'num_hidden_v', str(num_hidd))

        config.set('Example', 'name', exmaple)
        config.set('Example', 'repeat_time', str(repeat_time))

        lr0_u, lr0_v, lr0_rho = lr_func(dimx)
        config.set('Optimizer', 'lr0_u', str(lr0_u))
        config.set('Optimizer', 'lr0_v', str(lr0_v))
        config.set('Optimizer', 'lr0_rho', str(lr0_rho))

        max_it = maxit_func(dimx)
        config.set('Training', 'num_dt', str(num_dt))
        config.set('Training', 'max_iter', str(max_it))
        config.set('Training', 'batsize', str(bsz))
        idx_str = str(idx).zfill(length)
        sav_name = f'{idx_str}_{exmaple}_d{dimx}_W{wid}_H{num_hidd}_N{num_dt}'
        sav_config(config, f'{TASK_PATH}/{sav_name}')
        idx = idx + 1


def main():
    
    maxit_func = maxit_long
    # maxit_func = lambda _: 3000
    # maxit_func = maxit_default
    # maxit_func = lambda d: 6000
    width_func = lambda d: d + 100
    # width_func = lambda d: 256
    num_hidd = 6

    lr_func = ev_lrfunc

    repeat_time = 5
    dimx_list = [10, 20, 100, 1000, 2000]
    numdt_list = [8, 12, 18, 27, 40, 60, 90]
    # numdt_list = [int(8 * (128 / 8)**(i / 10)) for i in range(1, 11)]
    # dimx_list = [100, 1000]
    exa_list = [
        'EVP',
        # 'Counter',
        # 'HJB1',
        # 'HJB2bPba', 'HJB2bPbb', 'HJB2bPbc', 'HJB2bPbd', 'HJB2bPbe', 'HJB2bPbf', 'HJB2bPbg'
        # 'HJB1ShiftTarget2'
        # 'HJB1Onep',
        # 'HJB2b1p',
        # 'HJB2c1p',
        # 'HJB2b',
        # 'HJB0b',
        # 'HJB0c',
        # 'HJB2b',
        # 'HJB2c',
        # 'HJB1ShiftTarget',
        # 'HJB1ShiftTarget2',
        # 'HJB2bExtX0', 'HJB2bExtX0a', 'HJB2bExtX0b', 'HJB2bExtX0c'
    ]
    
    make_task(
        dimx_list,
        exa_list,
        numdt_list,
        num_hidd=num_hidd,
        maxit_func=maxit_func,
        width_func=width_func,
        repeat_time=repeat_time,
        lr_func=lr_func,
    )


if __name__ == "__main__":
    main()
