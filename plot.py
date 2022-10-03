import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, RocCurveDisplay
from utils import get_virtualprice, get_y_curve, get_dy_curve
# import pandas as pd
# from datetime import datetime

plt.ion()


def plot_results_confusion_matrix(Y_valid, Y_pred, Y_prob,
                                  threshold, learner, win):
    """
    """
    y_valid_cat = np.concatenate(Y_valid)
    y_pred_cat = np.concatenate(Y_pred)
    y_prob_cat = np.concatenate(Y_prob)
    f1 = f1_score(y_valid_cat, y_pred_cat)
    acc = (y_valid_cat == y_pred_cat).mean()

    cm = confusion_matrix(y_valid_cat, y_pred_cat)

    fig = plt.figure(figsize=(14*.8, 4.8*.8))
    ax = fig.add_subplot(121)
    im = ax.imshow(cm)
    for r in [0, 1]:
        for c in [0, 1]:
            if cm[r, c] < 100:
                color = 'w'
            else:
                color = 'k'
            ax.text(r, c, str(cm[r, c]), c=color)

    ax.set_xlabel('Predicted depeg')
    ax.set_ylabel('True depeg')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    fig.colorbar(im)
    # plot_confusion_matrix(cls, X_valid, y_valid, ax=ax)
    ax.set_title(f'$F_1$: {f1:1.3f}, accuracy: {acc:1.3f}')

    ax = fig.add_subplot(122)
    rd = RocCurveDisplay.from_predictions(y_valid_cat, y_prob_cat[:, 1], ax=ax)
    # color = rd.line_.get_color()
    ax.fill_between(rd.line_.get_xdata(), rd.line_.get_ydata(),
                    color=rd.line_.get_color(), alpha=0.15)
    ax.plot([0, 1], [0, 1], ':k')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    ax.set_title(f'ROC-curve $AUC$: {rd.roc_auc:1.3f}')
    ax.get_legend().remove()

    fig.suptitle(f'Depeg threshold: {threshold}% learner: {learner}, lag: {win}')

    fig.savefig(f'fig/depeg_confuse-error_thresh-{threshold}pct.png')


def plot_predictions(Y_valid, Y_pred, Y_prob,
                     threshold, learner, win, pool_names=[]):
    """
    """
    y_valid_cat = np.concatenate(Y_valid)
    y_pred_cat = np.concatenate(Y_pred)
    f1 = f1_score(y_valid_cat, y_pred_cat)
    acc = (y_valid_cat == y_pred_cat).mean()
    color_hits = 'forestgreen'
    color_misses = 'orangered'

    fig = plt.figure(figsize=[10,  6])

    for i, y in enumerate(Y_pred):
        ax = fig.add_subplot(len(Y_pred), 1, 1 + i)
        x = np.arange(len(y))
        hits = Y_valid[i] == Y_pred[i]
        misses = Y_valid[i] != Y_pred[i]
        ax.plot([x[0], x[-1]], [.5, .5], ':', c=[0]*3)
        ax.plot([x[0], x[-1]], [.0, .0], ':', c=[0.]*3)
        ax.plot([x[0], x[-1]], [1, 1], ':', c=[0.]*3)
        ax.plot(x, Y_valid[i], '-', c=[.7]*3, label='True depeg')
        # ax.plot(x-.1, y+.1, '.:r', label='Predicted depeg')
        ax.plot(x, Y_prob[i][:, 1], '-', c=[.3] * 3,
                label='Predicted $P_{depeg}$')
        ax.fill_between(x, Y_prob[i][:, 1], color='k', alpha=0.15)

        ax.plot(x[hits], Y_valid[i][hits], '.', mfc=color_hits,
                mec=color_hits, label='Correct predictions')
        ax.plot(x[misses], Y_valid[i][misses], '.', mfc=color_misses,
                mec=color_misses, label='Wrong predictions')

        ax.set_yticks([0, 1])
        ax.set_xlim([-1, len(Y_valid[i])])
        ax.set_yticklabels(['peg', 'depeg'])
        ax.set_ylim([-.1, 1.1])
        if i < (len(Y_pred) - 1):
            ax.set_xticklabels([])
        ax2 = ax.twinx()
        ax2.set_yticks([0, .5, 1])
        # ax.set_yticklabels([0, 0.5, 1])
        ax2.set_ylabel('Predicted $P_{depeg}$')
        ax2.set_ylim([-.1, 1.1])
        try:
            ax2.set_title(f'Pool: {pool_names[i]}')
        except IndexError:
            pass

    ax.legend()
    ax.set_xlabel('Time (days)')
    fig.suptitle(f'$F_1$: {f1:1.3f}, accuracy: {acc:1.3f}, '
                 f'depeg threshold: {threshold}%, learner: {learner}, lag: {win}')
    fig.savefig(f'fig/depeg_predictions_thresh-{threshold}pct.png')


def plot_curve(A_min=1):
    """
    """
    # A_range = np.exp(np.linspace(np.log(A_min), np.log(A_max), n_A))

    fig1 = plt.figure(figsize=(5.12, 4.2))
    ax = fig1.add_subplot(111)
    ratio = .2
    total = 2e6
    A = 16
    x1, y1 = total * ratio, total * (1 - ratio)
    x_range, y_curve = get_y_curve(A, x1, y1)
    dy = get_dy_curve(A, x1, y1)
    y0 = y1 - dy * x1
    vprice = round(get_virtualprice(A, x1, y1), 2)
    color = plt.cm.bone(0.6)

    ax.plot([x1, x1], [0, y1], color=[.75]*3)
    ax.plot([0, x1], [y1, y1], color=[.75]*3)
    ax.plot(x_range, y_curve, color=color,
            label=f'A: {A}, $x_1$: {int(100*ratio)}%, $x_1$ price: {vprice}')
    ax.plot([0.1, 2e6], [0.1*dy + y0, 2e6*dy + y0], ls=':', color=color)
    ax.plot(x1, y1, marker='o', ms=5, mfc='tab:orange', mec='tab:orange')

    ax.set_xlim((0, .3*1e7))
    ax.set_ylim((0, .3*1e7))
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.set_xlabel('Number of $x_1$ in pool')
    ax.set_ylabel('Number of $y_1$ in pool')
    ax.set_title('The token price depends on\nthe token ratio and the A parameter')

    fig1.savefig('fig/single_curve.png')

    fig2 = plt.figure(figsize=(9.6, 4.2))

    ax = fig2.add_subplot(121)
    A_range = 4 ** (A_min * np.arange(6))
    nline = A_range.shape[0]
    ratio = .2
    total = 2e6
    x1, y1 = total * ratio, total * (1 - ratio)

    for i, A in enumerate(A_range[::-1]):

        x_range, y_curve = get_y_curve(A, x1, y1)

        color = plt.cm.bone(0.8 * (i + 1) / (nline+1))
        dy = get_dy_curve(A, x1, y1)
        vprice = round(get_virtualprice(A, x1, y1), 2)
        ax.plot(x_range, y_curve, color=color,
                label=f'A: {A}, $x_1$ price: {vprice}')
        y0 = y1 - dy * x1
        ax.plot([0.1, 2e6], [0.1*dy + y0, 2e6*dy + y0], ls=':', color=color)
        ax.plot(x1, y1, marker='o', ms=5, mfc='tab:orange', mec='tab:orange')

    ax.set_xlim((0, .25*1e7))
    ax.set_ylim((0, .25*1e7))
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.set_xlabel('Number of $x_1$ in pool')
    ax.set_ylabel('Number of $y_1$ in pool')
    ax.set_title(f'Varying the A parameter; $x_1$: {int(100*ratio)}%')

    ax = fig2.add_subplot(122)
    # x1, y1 = 1e6, 1e6
    # x = np.linspace(x1*.0001, x1, n_A)
    # ratios = np.linspace(0.5, .98, 6)
    ratios = [0.5, 0.75, 0.85, 0.9, 0.95, 0.99]
    nline = len(ratios)
    total = 1e6 + 1e6
    A_contract = 4
    for i, ratio in enumerate(ratios):
        y1 = total * ratio
        x1 = total - y1

        color = plt.cm.bone(0.8 * (i + 1) / (nline+1))

        x_range, y_curve = get_y_curve(A_contract, x1, y1)
        dy = get_dy_curve(A_contract, x1, y1)
        vprice = round(get_virtualprice(A_contract, x1, y1), 2)

        ax.plot(x_range, y_curve, color=color,
                label=f'$x_1$: {int(100*(1-ratio))}%, $x_1$ price: {vprice}')
        y0 = y1 - dy * x1
        ax.plot([0.1, 2e6], [0.1*dy + y0, 2e6*dy + y0], ls=':', color=color)
        ax.plot(x1, y1, marker='o', ms=5, mfc='tab:orange', mec='tab:orange')
        # get_y(x[i], A, x1, y1)

    ax.set_xlim((0, .25*1e7))
    ax.set_ylim((0, .25*1e7))
    ax.set_xlabel('Number of $x_1$ in pool')
    ax.set_ylabel('Number of $y_1$ in pool')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.set_title(f'Varying token ratios in the pool; A: {A_contract}')

    fig2.savefig('fig/curves_A-tokRatio.png')


def plot_vprice_sim(A_min=1, A_max=1024, n_steps=100):
    """
    """
    A_range = np.exp(np.linspace(np.log(A_min), np.log(A_max), n_steps))
    n_A = A_range.shape[0]
    total = 2e7
    x1_range = np.linspace(0.5, 0.999, n_steps) * total
    y1_range = total - x1_range
    n_ratio = x1_range.shape[0]

    ratios = y1_range / (x1_range + y1_range)
    x, y = np.meshgrid(100 * ratios, A_range)

    vprices = np.zeros((n_A, n_ratio))

    for i, A in enumerate(A_range):
        for j, x1 in enumerate(x1_range):
            vprice = get_virtualprice(A, x1, y1_range[j])
            vprices[i, j] = vprice

    # vprices[vprices < 0.1] = np.nan
    vprices[vprices < 0.1] = .1
    Z = np.ma.array(vprices)

    fig = plt.figure(figsize=(5.12, 4.2))
    ax = fig.add_subplot(111)
    CS = ax.contourf(x, y, Z, 20, cmap=plt.cm.bone,
                     origin='lower', vmin=.1, vmax=1)
    cbar = fig.colorbar(CS, ticks=CS.levels[::2])
    cbar.add_lines([.95], colors=['r'], linewidths=[1])
    # cbar.set_clim([0.1, 1])
    cbar.set_label('Virtual price')
    CS = ax.contour(CS, levels=[.95], colors='r', origin='lower')

    # cbar.set_ticks(CS.levels)
    ax.set_ylabel('A')
    ax.set_xlabel('Percentage of $x_1$')
    xtcklbls = []
    for xt in ax.get_xticks():
        xtcklbls.append(f'{int(xt)}%')
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(xtcklbls)
    ax.set_title('Virtual price:\neffects of A and the token ratio in the pool')

    fig.savefig('fig/A-tokRatio_vprice.png')


def plot_sETH_ETH(df):

    fig = plt.figure(figsize=(15, 8))
    # ax = fig.add_subplot(211)
    ax = fig.add_axes((.1, .63, .8, .3))
    l0 = df['ratio'].plot(ax=ax, label='stETH/ETH')
    xlim = ax.get_xlim()
    l1 = ax.plot(xlim, [1, 1], ':k', label='Optimal stETH/ETH')
    l2 = df['price'].plot(ax=ax, color='tab:red', label='virtual price (ETH)')
    ax.set_xlim(xlim)
    ax.set_ylabel('stETH/ETH')
    ax.set_xlabel('')
    ax.legend(loc=2)
    ax = ax.twinx()
    l3 = df['A'].plot(ax=ax, color='tab:orange', label='A')
    ax.set_ylabel('A')
    ax.set_xlabel('')
    ax.set_xticks([])
    plt.legend(loc='upper right')

    ax = fig.add_axes((.1, .39, .8, .2))
    # l0 = df['ratio'].plot(ax=ax, label='stETH/ETH')
    df['price'].plot(ax=ax, color='tab:red', label='virtual price (ETH)')
    xlim = ax.get_xlim()
    ax.plot(xlim, [1, 1], '-', c=[.5]*3, label='Optimal price')
    ax.set_xlim(xlim)
    ax.set_ylim((.9994, 1.0005))
    ax.set_ylabel('virtual price (ETH)')
    ax.set_xlabel('')
    ax.legend(loc=2)
    ax = ax.twinx()
    df['A'].plot(ax=ax, color='tab:orange', label='A')
    ax.set_ylabel('A')
    ax.set_xlabel('')
    ax.legend(loc='upper right')

    # ax = fig.add_subplot(223)
    ax = fig.add_axes((.1, .07, .37, .25))
    ids = np.logical_and(df.index > '2021-02-19', df.index < '2021-03-01')
    df.loc[ids, 'ratio'].plot(ax=ax)
    df.loc[ids, 'price'].plot(ax=ax, color='tab:red')
    ax.plot(xlim, [1, 1], ':k')
    ax.set_xlim(xlim)
    ax.set_ylabel('stETH/ETH')
    ax = ax.twinx()
    df.loc[ids, 'A'].plot(ax=ax, color='tab:orange')
    ax.set_xlabel('')

    # ax = fig.add_subplot(224)
    ax = fig.add_axes((.53, .07, .37, .25))
    ids = np.logical_and(df.index > '2021-05-13', df.index < '2021-05-23')
    df.loc[ids, 'ratio'].plot(ax=ax)
    df.loc[ids, 'price'].plot(ax=ax, color='tab:red')
    ax.plot(xlim, [1, 1], ':k')
    ax.set_xlim(xlim)
    ax = ax.twinx()
    df.loc[ids, 'A'].plot(ax=ax, color='tab:orange')
    ax.set_ylabel('A')
    ax.set_xlabel('')

    fig.suptitle('Curve-sETH-ETH')

    fig.savefig('fig/Curve-sETH-ETH_raio-A.png')


def plot_3Pool(df):

    fig = plt.figure(figsize=(15, 8))
    # ax = fig.add_subplot(211)
    ax = fig.add_axes((.1, .63, .8, .3))
    df['ratio_USDT/USDC'].plot(ax=ax, color='tab:blue', label='USDT/USDC')
    df['ratio_DAI/USDC'].plot(ax=ax, color='tab:green', label='DAI/USDC')
    df['ratio_DAI/USDT'].plot(ax=ax, color='tab:orange', label='DAI/USDT')
    xlim = ax.get_xlim()
    ax.plot(xlim, [1, 1], ':k', label='Optimal ratios')
    df['price'].plot(ax=ax, color='tab:red', label='virtual price')
    ax.set_xlim(xlim)
    ax.set_ylabel('ratios')
    ax.set_xlabel('')
    ax.legend(loc=2)
    ax = ax.twinx()
    df['A'].plot(ax=ax, color='m', label='A')
    ax.set_ylabel('A')
    ax.set_xlabel('')
    ax.set_xticks([])
    plt.legend(loc='upper right')

    ax = fig.add_axes((.1, .39, .8, .2))
    # l0 = df['ratio'].plot(ax=ax, label='stETH/ETH')
    df['price'].plot(ax=ax, color='tab:red', label='virtual price')
    xlim = ax.get_xlim()
    ax.plot(xlim, [1, 1], '-', c=[.5]*3, label='Optimal price')
    ax.set_xlim(xlim)
    ax.set_ylim((.99, 1.025))
    ax.set_ylabel('virtual price (ETH)')
    ax.set_xlabel('')
    ax.legend(loc=2)
    ax = ax.twinx()
    df['A'].plot(ax=ax, color='m', label='A')
    ax.set_ylabel('A')
    ax.set_xlabel('')
    ax.legend(loc='upper right')

    # ax = fig.add_subplot(223)
    ax = fig.add_axes((.1, .07, .37, .25))
    ids = np.logical_and(df.index > '2021-04-15', df.index < '2021-05-30')
    df.loc[ids, 'ratio_USDT/USDC'].plot(ax=ax,
                                        color='tab:blue',
                                        label='USDT/USDC')
    df.loc[ids, 'ratio_DAI/USDC'].plot(ax=ax,
                                       color='tab:green',
                                       label='DAI/USDC')
    df.loc[ids, 'ratio_DAI/USDT'].plot(ax=ax,
                                       color='tab:orange',
                                       label='DAI/USDT')
    df.loc[ids, 'price'].plot(ax=ax, color='tab:red')
    xlim = ax.get_xlim()
    ax.plot(xlim, [1, 1], ':k')
    # ax.set_ylim((.975, 1.025))
    ax.set_xlim(xlim)
    ax.set_ylabel('ratios')
    ax = ax.twinx()
    df.loc[ids, 'A'].plot(ax=ax, color='m')
    ax.set_xlabel('')

    # ax = fig.add_subplot(224)
    ax = fig.add_axes((.53, .07, .37, .25))
    ids = np.logical_and(df.index > '2022-04-20', df.index < '2022-06-08')
    df.loc[ids, 'ratio_USDT/USDC'].plot(ax=ax,
                                        color='tab:blue',
                                        label='USDT/USDC')
    df.loc[ids, 'ratio_DAI/USDC'].plot(ax=ax,
                                       color='tab:green',
                                       label='DAI/USDC')
    df.loc[ids, 'ratio_DAI/USDT'].plot(ax=ax,
                                       color='tab:orange',
                                       label='DAI/USDT')
    df.loc[ids, 'price'].plot(ax=ax, color='tab:red')
    xlim = ax.get_xlim()
    ax.plot(xlim, [1, 1], ':k')
    ax.set_xlim(xlim)
    # ax.set_ylim((.975, 1.025))
    ax = ax.twinx()
    df.loc[ids, 'A'].plot(ax=ax, color='tab:orange')
    ax.set_ylabel('A')
    ax.set_xlabel('')

    fig.suptitle('Curve-3Pool')

    fig.savefig('fig/Curve-3Pool.png')


def plot_degenbox(df):
    """
    """

    fig = plt.figure(figsize=(15, 8))
    # ax = fig.add_subplot(211)
    ax = fig.add_axes((.1, .63, .8, .3))
    df['ratio'].plot(ax=ax, color='tab:blue', label='UST/MIM')
    xlim = ax.get_xlim()
    ax.plot(xlim, [1, 1], ':k', label='Optimal ratios')
    df['price'].plot(ax=ax, color='tab:red', label='virtual price')
    ax.set_xlim(xlim)
    ax.set_ylabel('ratios')
    ax.set_xlabel('')
    ax.legend(loc=2)
    ax = ax.twinx()
    df['A'].plot(ax=ax, color='m', label='A')
    ax.set_ylabel('A')
    ax.set_xlabel('')
    ax.set_xticks([])
    plt.legend(loc='upper right')

    ax = fig.add_axes((.1, .39, .8, .2))
    # l0 = df['ratio'].plot(ax=ax, label='stETH/ETH')
    df['price'].plot(ax=ax, color='tab:red', label='virtual price')
    xlim = ax.get_xlim()
    ax.plot(xlim, [1, 1], ':', c=[.5]*3, label='Optimal price')
    ax.set_xlim(xlim)
    ax.set_ylim((.3, 1.1))
    ax.set_ylabel('virtual price (MIM)')
    ax.set_xlabel('')
    ax.legend(loc=2)
    ax = ax.twinx()
    df['A'].plot(ax=ax, color='m', label='A')
    ax.set_ylabel('A')
    ax.set_xlabel('')
    ax.legend(loc='upper right')

    # ax = fig.add_subplot(223)
    ax = fig.add_axes((.1, .07, .37, .25))
    ids = np.logical_and(df.index > '2021-12-15', df.index < '2022-03-01')
    df.loc[ids, 'ratio'].plot(ax=ax, color='tab:blue', label='UST/MIM')
    df.loc[ids, 'price'].plot(ax=ax, color='tab:red')
    xlim = ax.get_xlim()
    ax.plot(xlim, [1, 1], ':k')
    # ax.set_ylim((.975, 1.025))
    ax.set_xlim(xlim)
    ax.set_ylabel('ratios')
    ax = ax.twinx()
    df.loc[ids, 'A'].plot(ax=ax, color='m')
    ax.set_xlabel('')

    # ax = fig.add_subplot(224)
    ax = fig.add_axes((.53, .07, .37, .25))
    ids = np.logical_and(df.index > '2022-05-01', df.index < '2022-06-28')
    df.loc[ids, 'ratio'].plot(ax=ax, color='tab:blue', label='UST/MIM')
    df.loc[ids, 'price'].plot(ax=ax, color='tab:red')
    xlim = ax.get_xlim()
    ax.plot(xlim, [1, 1], ':k')
    ax.set_xlim(xlim)
    # ax.set_ylim((.975, 1.025))
    ax = ax.twinx()
    df.loc[ids, 'A'].plot(ax=ax, color='m')
    ax.set_ylabel('A')
    ax.set_xlabel('')

    fig.suptitle('Curve-Degenbox')

    fig.savefig('fig/Curve-Degenbox.png')


def plot_usdn(df):
    """
    """

    fig = plt.figure(figsize=(15, 8))
    # ax = fig.add_subplot(211)
    ax = fig.add_axes((.1, .63, .8, .3))
    df['ratio'].plot(ax=ax, color='tab:blue', label='USDN/3CRV')
    xlim = ax.get_xlim()
    ax.plot(xlim, [1, 1], ':k', label='Optimal ratios')
    df['price'].plot(ax=ax, color='tab:red', label='virtual price')
    ax.set_xlim(xlim)
    ax.set_ylabel('ratios')
    ax.set_xlabel('')
    ax.legend(loc=2)
    ax = ax.twinx()
    df['A'].plot(ax=ax, color='m', label='A')
    ax.set_ylabel('A')
    ax.set_xlabel('')
    ax.set_xticks([])
    plt.legend(loc='upper right')

    ax = fig.add_axes((.1, .39, .8, .2))
    # l0 = df['ratio'].plot(ax=ax, label='stETH/ETH')
    df['price'].plot(ax=ax, color='tab:red', label='virtual price')
    xlim = ax.get_xlim()
    ax.plot(xlim, [1, 1], ':', c=[.5]*3, label='Optimal price')
    ax.set_xlim(xlim)
    ax.set_ylim((.995, 1.02))
    ax.set_ylabel('virtual price (USDN)')
    ax.set_xlabel('')
    ax.legend(loc=2)
    ax = ax.twinx()
    df['A'].plot(ax=ax, color='m', label='A')
    ax.set_ylabel('A')
    ax.set_xlabel('')
    ax.legend(loc='upper right')

    # ax = fig.add_subplot(223)
    ax = fig.add_axes((.1, .07, .37, .25))
    ids = np.logical_and(df.index > '2020-12-11', df.index < '2021-02-27')
    df.loc[ids, 'ratio'].plot(ax=ax, color='tab:blue', label='UST/MIM')
    df.loc[ids, 'price'].plot(ax=ax, color='tab:red')
    xlim = ax.get_xlim()
    ax.plot(xlim, [1, 1], ':k')
    # ax.set_ylim((.975, 1.025))
    ax.set_xlim(xlim)
    ax.set_ylabel('ratios')
    ax = ax.twinx()
    df.loc[ids, 'A'].plot(ax=ax, color='m')
    ax.set_xlabel('')

    # ax = fig.add_subplot(224)
    ax = fig.add_axes((.53, .07, .37, .25))
    ids = np.logical_and(df.index > '2022-03-25', df.index < '2022-07-01')
    df.loc[ids, 'ratio'].plot(ax=ax, color='tab:blue', label='UST/MIM')
    df.loc[ids, 'price'].plot(ax=ax, color='tab:red')
    xlim = ax.get_xlim()
    ax.plot(xlim, [1, 1], ':k')
    ax.set_xlim(xlim)
    # ax.set_ylim((.975, 1.025))
    ax = ax.twinx()
    df.loc[ids, 'A'].plot(ax=ax, color='m')
    ax.set_ylabel('A')
    ax.set_xlabel('')

    fig.suptitle('Curve-USDN-3CRV')

    fig.savefig('fig/Curve-USDN-3CRV.png')
