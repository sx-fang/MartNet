from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from martnetdf import MartNet


def tensor2ndarray(tensor):
    return tensor.detach().cpu().numpy()


def plot_hist(sav_name,
              x_arr_list,
              y_arr_list,
              std_arr_list=None,
              lable_list=None,
              color_list=None,
              xlabel=None,
              ylabel=None,
              alpha=1.0,
              yscal='log',
              linewidth=None,
              ylim0=None,
              ylim1=None,
              two_side_std=False):
    num_arrs = len(x_arr_list)

    if lable_list is None:
        lable_list = [None] * num_arrs
    if all(lable_list) is False:
        plot_legend = False
    else:
        plot_legend = True
    if color_list is None:
        c_list = plt.get_cmap('tab10')(np.arange(num_arrs))
    if std_arr_list is None:
        std_arr_list = [None] * num_arrs

    plt.yscale(yscal)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for x_arr, y_arr, std_arr, label, color in zip(x_arr_list, y_arr_list,
                                                   std_arr_list, lable_list,
                                                   c_list):
        plt.plot(x_arr,
                 y_arr,
                 label=label,
                 alpha=alpha,
                 color=color,
                 linewidth=linewidth)
        if std_arr is not None:
            yup_arr = y_arr + 2 * std_arr
            if two_side_std is True:
                ylow_arr = y_arr - 2 * std_arr
            else:
                ylow_arr = y_arr
            plt.fill_between(
                x_arr,
                ylow_arr,
                yup_arr,
                facecolor=color,
                alpha=0.5 * alpha,
            )
    plt.ylim(ylim0, ylim1)
    plt.tight_layout()
    if plot_legend:
        plt.legend()
    Path(sav_name).parent.mkdir(exist_ok=True, parents=True)
    plt.grid(True)
    plt.savefig(sav_name)
    plt.close()


def plot_l1linf_error(hist_df, sav_path):
    l1err = hist_df['rel_l1err']
    linf_err = hist_df['rel_linferr']
    plot_hist(sav_path + 'error_hist.pdf', [hist_df['it'].to_numpy()] * 2,
              (linf_err, l1err),
              lable_list=(r"Rel. $L^{\infty}$-error", "Rel. $L^1$-error"),
              alpha=1.0)


def plot_linf_error(hist_df, sav_path):
    linf_err = hist_df['rel_linferr']
    plot_hist(sav_path + 'error_hist.pdf', (hist_df['it'].to_numpy(), ),
              (linf_err, ),
              lable_list=None,
              alpha=1.0)


def ylim_for_g1loss(g1_hist):
    y_median = np.median(g1_hist)
    iqr = np.quantile(g1_hist, 0.75) - np.quantile(g1_hist, 0.25)
    ylim0 = y_median - 10 * max(iqr, 0.05)
    ylim1 = y_median + 10 * max(iqr, 0.05)
    return ylim0, ylim1


def save_results(martnet: MartNet, hist_dict, sav_path):
    Path(sav_path).parent.mkdir(exist_ok=True, parents=True)

    hist_df = pd.DataFrame(hist_dict)
    hist_df.to_csv(f'{sav_path}log.csv', index=False)
    it_arr = np.array(hist_df['it'])

    problem = martnet.problem
    problem.produce_results(martnet.vnn, sav_path)
    plot_hist(sav_path + 'lossmart_hist.pdf', (it_arr, ),
              (hist_df['mart_loss'], ))

    g1_hist = hist_df['g1_loss'].to_numpy()
    ylim0, ylim1 = ylim_for_g1loss(g1_hist)
    plot_hist(sav_path + 'g1_loss.pdf', (it_arr, ), (g1_hist, ),
              yscal='linear',
              ylim0=ylim0,
              ylim1=ylim1)

    if ("cost" in hist_dict) and ("mean_vtrue_t0" in hist_dict):
        it_cost = hist_df[['it', 'cost', 'mean_vtrue_t0']].dropna()
        if len(it_cost) > 0:
            plot_hist(sav_path + 'cost_hist.pdf',
                      (it_cost['it'], it_cost['it']),
                      (it_cost['cost'], it_cost['mean_vtrue_t0']),
                      lable_list=(r'$J(u_{\theta})$', '$J(u^*)$'),
                      yscal='linear')

    if ("rel_l1err" in hist_dict) and ("rel_linferr" in hist_dict):
        plot_l1linf_error(hist_df, sav_path)
    elif "rel_linferr" in hist_dict:
        plot_linf_error(hist_df, sav_path)

    if "ev_error" in hist_dict:
        plot_hist(sav_path + 'ev_error_hist.pdf', (hist_df['it'].to_numpy(), ),
                  (hist_df['ev_error'], ),
                  lable_list=None,
                  alpha=1.0)


def summary_hist(hist_list) -> pd.DataFrame:

    df_list = [pd.DataFrame(hist) for hist in hist_list]
    hist_df = pd.concat(df_list,
                        axis=0,
                        keys=range(len(df_list)),
                        names=['round', 'index'])
    hist_gp = hist_df.groupby('it')
    mean_df = hist_gp.mean()
    std_df = hist_gp.std()
    summ_df = pd.concat([mean_df, std_df], axis=1, keys=['mean', 'std'])
    summ_df = summ_df.swaplevel(0, 1, axis=1).sort_index(axis=1)
    return summ_df


def plot_summary(summ_df: pd.DataFrame, sav_name: str):
    it_arr = np.array(summ_df.index)
    plot_hist(sav_name + 'maloss_mean_std.pdf', (it_arr, ),
              (summ_df['mart_loss', 'mean'], ),
              std_arr_list=(summ_df['mart_loss', 'std'], ),
              lable_list=None,
              yscal='log',
              alpha=1.0,
              two_side_std=False)

    mean_g1loss = summ_df['g1_loss', 'mean'].to_numpy()
    ylim0, ylim1 = ylim_for_g1loss(mean_g1loss)
    plot_hist(
        sav_name + 'g1loss_mean_std.pdf',
        (it_arr, ),
        (mean_g1loss, ),
        std_arr_list=(summ_df['g1_loss', 'std'], ),
        lable_list=None,
        yscal='linear',
        alpha=1.0,
        two_side_std=False,
        ylim0=ylim0,
        ylim1=ylim1,
    )

    record_l1err = ('rel_l1err', 'mean') in summ_df.columns \
        and ('rel_l1err', 'std') in summ_df.columns
    record_linferr = ('rel_linferr', 'mean') in summ_df.columns \
        and ('rel_linferr', 'std') in summ_df.columns
    if record_l1err:
        mean_l1err = summ_df['rel_l1err', 'mean'].to_numpy()
        std_l1err = summ_df['rel_l1err', 'std'].to_numpy()
    if record_linferr:
        mean_linferr = summ_df['rel_linferr', 'mean'].to_numpy()
        std_linferr = summ_df['rel_linferr', 'std'].to_numpy()

    if record_l1err and record_linferr:
        plot_hist(sav_name + 'error_mean_std.pdf', (it_arr, it_arr),
                  (mean_linferr, mean_l1err),
                  std_arr_list=(std_linferr, std_l1err),
                  lable_list=(r"Rel. $L^{\infty}$-error", "Rel. $L^1$-error"),
                  yscal='log',
                  alpha=1.0,
                  two_side_std=False)
    elif record_l1err:
        plot_hist(
            sav_name + 'error_mean_std.pdf',
            (it_arr, ),
            (mean_l1err, ),
            std_arr_list=(std_l1err, ),
            lable_list=None,
            yscal='log',
            alpha=1.0,
            two_side_std=False,
        )
    elif record_linferr:
        plot_hist(
            sav_name + 'error_mean_std.pdf',
            (it_arr, ),
            (mean_linferr, ),
            std_arr_list=(std_linferr, ),
            lable_list=None,
            yscal='log',
            alpha=1.0,
            two_side_std=False,
        )

    if (('rel_everr', 'mean') in summ_df.columns) \
            and (('rel_everr', 'mean') in summ_df.columns):
        mean_everr = summ_df['rel_everr', 'mean'].to_numpy()
        std_everr = summ_df['rel_everr', 'std'].to_numpy()
        plot_hist(
            sav_name + 'ev_error_mean_std.pdf',
            (it_arr, ),
            (mean_everr, ),
            std_arr_list=(std_everr, ),
            lable_list=None,
            yscal='log',
            alpha=1.0,
            two_side_std=False,
        )

    if (('cost', 'mean') in summ_df.columns) \
            and (('mean_vtrue_t0', 'mean') in summ_df.columns):
        cost_df = summ_df[[('cost', 'mean'), ('cost', 'std'),
                           ('mean_vtrue_t0', 'mean'),
                           ('mean_vtrue_t0', 'std')]].dropna()
        if len(cost_df) > 0:
            it_cost = cost_df.index.to_numpy()
            mean_cost = cost_df['cost', 'mean'].to_numpy()
            std_cost = cost_df['cost', 'std'].to_numpy()
            mean_vtrue_t0 = cost_df['mean_vtrue_t0', 'mean'].to_numpy()
            std_vtrue_t0 = cost_df['mean_vtrue_t0', 'std'].to_numpy()
            plot_hist(
                sav_name + 'cost_mean_std.pdf',
                (it_cost, it_cost),
                (mean_cost, mean_vtrue_t0),
                std_arr_list=(std_cost, std_vtrue_t0),
                lable_list=(r'$J(u_{\theta})$', '$J(u^*)$'),
                yscal='linear',
            )
