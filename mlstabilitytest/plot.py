#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:51:51 2019

@author: chrisbartel
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from compmatscipy.plotting_functions import set_rc_params, tableau_colors
from compmatscipy.handy_functions import read_json
from mlstabilitytest.mp_data.data import hullout, mp_LiMnTMO, smact_LiMnTMO
import os
import numpy as np
from scipy.stats import linregress
from prettytable import PrettyTable

this_dir, this_filename = os.path.split(__file__)
FIG_DIR = os.path.join(this_dir, 'figures')
def get_results(training_prop, experiment, model):
    return read_json(os.path.join(this_dir, 
                                  'ml_data', 
                                  training_prop, 
                                  experiment, 
                                  model,
                                  'ml_results.json'))
def get_compounds(experiment):
    if experiment == 'allMP':
        d = hullout()
        return sorted(list(d.keys()))
    elif experiment == 'LiMnTMO':
        d = mp_LiMnTMO()
        return sorted(list(d.keys()))
    elif experiment == 'smact':
        d1 = mp_LiMnTMO()
        d2 = smact_LiMnTMO()
        return sorted(list(set(list(d1.keys())+list(d2.keys()))))
    
def get_actual(prop, compounds):
    d = hullout()
    return [d[c][prop] for c in compounds]

def get_pred(training_prop, prop, experiment, model, compounds):
    results = get_results(training_prop, experiment, model)
    d = dict(zip(results['data']['formulas'], results['data'][prop]))
    return [d[c] for c in d]

def get_mae(actual, pred):
    return np.mean([abs(actual[i] - pred[i]) for i in range(len(actual))])
        
def ax_generic_scatter(x, y, 
                       alpha=0.1,
                       marker='o',
                       lw=0,
                       s=10,
                       colors='black',
                       edgecolors=None,
                       vmin=False,
                       vmax=False,
                       cmap=False,
                       cmap_values=False,
                       xticks=(True, (0, 1)),
                       yticks=(True, (0, 1)),
                       xlabel='x',
                       ylabel='y',
                       xlim=(0,1),
                       ylim=(0,1),
                       diag=('black', 1, '--')):
    if colors == 'cmap':
        cm = plt.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        ax = plt.scatter(x, y,
                         c=cmap_values,
                         cmap=cm,
                         norm=norm,
                         edgecolors=edgecolors,
                         alpha=alpha, 
                         marker=marker, 
                         lw=lw, 
                         s=s)
    else:
        if isinstance(colors, list):
            ax = plt.scatter(x, y, 
                             c=colors, 
                             edgecolors=edgecolors,
                             alpha=alpha, 
                             marker=marker, 
                             lw=lw, 
                             s=s)
        else:
            ax = plt.scatter(x, y, 
                             color=colors, 
                             edgecolor=edgecolors,
                             alpha=alpha, 
                             marker=marker, 
                             lw=lw, 
                             s=s)
    ax = plt.xticks(xticks[1])
    ax = plt.yticks(yticks[1])
    if not xticks[0]:
        ax = plt.gca().xaxis.set_ticklabels([])
    if not yticks[0]:
        ax = plt.gca().yaxis.set_ticklabels([])
    ax = plt.xlabel(xlabel)
    ax = plt.ylabel(ylabel)
    if diag:
        ax = plt.plot(xlim, xlim, color=diag[0], lw=diag[1], ls=diag[2])
    ax = plt.xlim(xlim)
    ax = plt.ylim(ylim)
    return ax

def ax_actual_vs_pred(actual, pred, prop, exp='allMP',
                      show_xticks=False, show_yticks=False,
                      show_xlabel=False, show_ylabel=False,
                      show_mae=False, show_model=False):
    
    x, y = pred, actual
    if exp == 'allMP':
        alpha = 0.1
    else:
        alpha = 0.4
    marker = 'o'
    lw = 0
    if exp == 'allMP':
        s = 10
    else:
        s = 20
    colors='cmap'
    edgecolors = None
    cmap = 'plasma_r'
    cmap_values = [abs(actual[i] - pred[i]) for i in range(len(actual))]
    vmin, vmax = 0, 1
    diag = ('black', 1, '--')
    if prop == 'Ef':
        if exp == 'allMP':
            ticks = (-4, -3, -2, -1, 0, 1)
        else:
            ticks = (-3, -2, -1)
    elif prop == 'Ed':
        ticks = (-1, 0, 1, 2)
    xtick_values, ytick_values = ticks, ticks
    
    if show_xlabel or show_ylabel:
        label_dH = r'$\Delta$'+r'$\it{H}$'
        label_units = r'$\/(\frac{eV}{atom})$'
        if prop == 'Ef':
            xlabel = label_dH+r'$_{f,pred}$'+label_units
            ylabel = xlabel.replace('pred', 'MP')
        elif prop == 'Ed':
            xlabel = label_dH+r'$_{d,pred}$'+label_units
            ylabel = xlabel.replace('pred', 'MP') 
    if not show_xlabel:
        xlabel = ''
    if not show_ylabel:
        ylabel = ''
            
    if prop == 'Ef':
        if exp == 'allMP':
            xlim = (min(actual), max(actual))
            spacer = 0.1
            xlim = (xlim[0]-spacer, xlim[1]+spacer)
        else:
            xlim = (-3, -1)
    elif prop == 'Ed':
        xlim = (-1, 2)
    ylim = xlim
    
    ax = ax_generic_scatter(x, y, 
                            alpha=alpha,
                            marker=marker,
                            lw=lw,
                            s=s,
                            colors=colors,
                            edgecolors=edgecolors,
                            vmin=vmin,
                            vmax=vmax,
                            cmap=cmap,
                            cmap_values=cmap_values,
                            xticks=(show_xticks, xtick_values),
                            yticks=(show_yticks, ytick_values),
                            xlabel=xlabel,
                            ylabel=ylabel,
                            xlim=xlim,
                            ylim=ylim,
                            diag=diag)
    
    if show_model:
        x_range = xlim[1] - xlim[0]
        x_offset, y_offset = 0.03, 0.15
        ax = plt.text(xlim[0]+x_offset*x_range, xlim[1]-y_offset*x_range, show_model)
    if show_mae:
        if prop == 'Ef':
            if exp == 'allMP':
                xpos, ypos = 1, -4.5
            else:
                xpos, ypos = -1.05, -2.9
        elif prop == 'Ed':
            xpos, ypos = 1.9, -0.95
        MAE = np.round(show_mae, 2)
        MAE_font = 15
        ax = plt.text(xpos, ypos, r'$MAE=%.2f\/\frac{eV}{atom}$' % MAE, 
                      fontsize=MAE_font,
                      horizontalalignment='right',
                      verticalalignment='bottom')
    
    ax = plt.xlim(xlim)
    ax = plt.ylim(ylim)
    return ax    

def add_colorbar(fig, label, ticks, 
                 cmap, vmin, vmax, position, 
                 label_size, tick_size, tick_len, tick_width,
                 orientation='vertical',
                 tick_pos=False):
    norm = plt.cm.colors.Normalize(vmin=vmin, vmax=vmax)
    cax = fig.add_axes(position)    
    cb = mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap, norm=norm, orientation=orientation)
    cb.set_label(label, fontsize=label_size)
    cb.set_ticks(ticks)
    cb.ax.set_yticklabels(ticks)
    cb.ax.tick_params(labelsize=tick_size, length=tick_len, width=tick_width)
    if tick_pos:
        cb.ax.yaxis.set_ticks_position(tick_pos)
    return fig 

def ax_hist_classification(training_prop, model, show_xlabel=False, show_ylabel=False, leg=False, show_model=True, show_yticks=False):
    experiment = 'allMP'
    prop = 'Ed'
    compounds = get_compounds(experiment)
    actual = get_actual(prop, compounds)
    pred = get_pred(training_prop, prop, experiment, model, compounds)
    results = get_results(training_prop, experiment, model)
    stats = results['stats']['Ed']['cl']['0']
    bins = 200
    xlim = (-0.2, 0.2)
    ylim = (0, 1200)
    colors = tableau_colors()
    alpha = 0.5
    xticks = (-0.2, -0.1, 0, 0.1, 0.2)
    yticks = (0, 400, 800, 1200)
    
    tp_indices = [i for i in range(len(actual)) 
                    if actual[i] <= 0 
                    if pred[i] <= 0]
    fp_indices = [i for i in range(len(actual)) 
                    if actual[i] > 0 
                    if pred[i] <= 0]
    tn_indices = [i for i in range(len(actual)) 
                    if actual[i] > 0 
                    if pred[i] > 0]
    fn_indices = [i for i in range(len(actual)) 
                    if actual[i] <= 0 
                    if pred[i] > 0]
    tp = [actual[i] for i in tp_indices]
    fp = [actual[i] for i in fp_indices]
    tn = [actual[i] for i in tn_indices]
    fn = [actual[i] for i in fn_indices]    
        
    ax = plt.hist(tp, 
                  bins=bins,
                  range=xlim,
                  color=colors['blue'],
                  alpha=alpha,
                  label='__nolegend__')
    ax = plt.hist(fn,
                  bins=bins,
                  range=xlim,
                  color=colors['red'],
                  alpha=alpha,
                  label='__nolegend__')
    ax = plt.hist(tn, 
                  bins=bins,
                  range=xlim,
                  color=colors['blue'],
                  alpha=alpha,
                  label='correct')
    ax = plt.hist(fp,
                  bins=bins,
                  range=xlim,
                  color=colors['red'],
                  alpha=alpha,
                  label='incorrect')
    
    ax = plt.plot([0, 0], [0,100000], color='black', lw=1.5, ls='--', alpha=alpha)
    
    xpos, ypos = 0.18, 1100
    fontsize=14
    ax = plt.text(xpos, ypos, 'Acc = %.2f\nF1 = %.2f\nFPR = %.2f' 
                      % (stats['scores']['accuracy'],
                         stats['scores']['f1'],
                         stats['scores']['fpr']),
                         fontsize=fontsize,
                         verticalalignment='top'
                         ,horizontalalignment='right') 
                      
    xpos, ypos = -0.19, 1150
    fontsize=20
    if show_model:
        ax = plt.text(xpos, ypos, model, verticalalignment='top')
    
    ax = plt.xticks(xticks)
    ax = plt.yticks(yticks)
    if show_xlabel:
        xlabel = r'$\Delta$'+r'$\it{H}$'+r'$_{d,MP}$'+r'$\/(\frac{eV}{atom})$'
    else:
        xlabel = ''
        ax = plt.gca().xaxis.set_ticklabels([])
    if show_ylabel:
        ylabel = 'Number of compounds'
    else:
        ylabel = ''
    if not show_yticks:
        ax = plt.gca().yaxis.set_ticklabels([])
    ax = plt.xlabel(xlabel)
    ax = plt.ylabel(ylabel)
    ax = plt.ylim(ylim)
    ax = plt.xlim(xlim)
    if leg:
        ax = plt.legend(loc=leg)
    
    return ax

def ax_LiMnTMO(training_prop, model, show_xlabel, show_ylabel, show_model=True):
    experiment = 'LiMnTMO'
    prop = 'Ed'
    compounds = get_compounds(experiment)
    actual = get_actual(prop, compounds)
    pred = get_pred(training_prop, prop, experiment, model, compounds)
    xticks_Ed = (-0.2, -0.1, 0, 0.1, 0.2)
    yticks_Ed = (-0.1, 0, 0.1, 0.2)
    xlim_Ed, ylim_Ed = (-0.2, 0.25), (-0.1, 0.25)
    alpha = 1
    s = 50
    c_stable = tableau_colors()['blue']
    c_unstable = tableau_colors()['red']
    ax = plt.plot([0, 0], ylim_Ed, ls='--', color='black', alpha=0.4)
    ax = plt.plot(xlim_Ed, [0, 0], ls='--', color='black', alpha=0.4)
    labels = []
    for i in range(len(actual)):
        if (actual[i] <= 0) and (pred[i] <= 0):
            label = 'tp'
        elif pred[i] <= 0:
            label = 'fp'
        elif (actual[i] > 0) and (pred[i] > 0):
            label = 'tn'
        else:
            label = 'fn'
        labels.append(label)

    data = {label : {'actual' : [actual[i] for i in range(len(labels)) if labels[i] == label],
                     'pred' : [pred[i] for i in range(len(labels)) if labels[i] == label]}
                for label in ['tp', 'fp', 'tn', 'fn']}
    alpha = 0.5
    for label in data:
        if label == 'tp':
            color = c_stable
            edgecolor = c_stable
            marker = 'o'
        elif label == 'tn':
            color = c_stable
            edgecolor = c_stable
            marker = 'o'
        elif label == 'fp':
            color = c_unstable
            edgecolor = c_unstable
            marker = '^'
        elif label == 'fn':
            color = c_unstable
            edgecolor = c_unstable
            marker = '^'
        pred = data[label]['pred']
        actual = data[label]['actual']                   
        ax = ax_generic_scatter(pred, actual, 
                                colors=color,
                                marker=marker,
                                s=s,
                                edgecolors=edgecolor,
                                xticks=(True, xticks_Ed), yticks=(True, yticks_Ed),
                                xlim=xlim_Ed, ylim=ylim_Ed, alpha=alpha, diag=False)
    Ed_font = 16
    
    ax = plt.text(-0.19, -0.09, 
                   'TP = %i' % (len(data['tp']['actual'])),
                   color=c_stable,
                   fontsize=Ed_font,
                   verticalalignment='bottom')
    ax = plt.text(-0.19, 0.22, 
                   'FP = %i' % (len(data['fp']['actual'])),
                   color=c_unstable,
                   verticalalignment='top',
                   fontsize=Ed_font)
    ax = plt.text(0.22, 0.22, 
                   'TN = %i' % (len(data['tn']['actual'])),
                   color=c_stable,
                   horizontalalignment='right',
                   verticalalignment='top',
                   fontsize=Ed_font)
    ax = plt.text(0.22, -0.09, 
                   'FN = %i' % (len(data['fn']['actual'])),
                   color=c_unstable,
                   horizontalalignment='right',
                   verticalalignment='bottom',
                   fontsize=Ed_font)
    
    if show_xlabel:
        ax = plt.xlabel(r'$\Delta$' + r'$\it{H}$' + r'$_{d,pred}$' + r'$\/(\frac{eV}{atom})$')
    else:
        ax = plt.xlabel('')
        ax = plt.gca().xaxis.set_ticklabels([])
    if show_ylabel:
        ax = plt.ylabel(r'$\Delta$' + r'$\it{H}$' + r'$_{d,MP}$' + r'$\/(\frac{eV}{atom})$')
    else:
        ax = plt.ylabel('')
        ax = plt.gca().yaxis.set_ticklabels([])
        
    if show_model:
        xpos = 0
        ypos = 0.27
        model_font = 20
        ax = plt.text(xpos, ypos,
                       model,
                       fontsize=model_font,
                       horizontalalignment='center',
                       verticalalignment='bottom')
    return ax
    
def make_fig1():
    fig = plt.figure(figsize=(12, 3.5))
    ax1 = plt.subplot(121)
    xticks = (0, 0.2, 0.4, 0.6, 0.8, 1)
    xlim = (0, 1)
    ylim = (-1, 0)
    yticks = (-1, -0.8, -0.6, -0.4, -0.2, 0)
    xlabel = r'$x\/in\/A_{1-x}B_x$'
    ylabel = r'$\Delta$' + r'$\it{H}$' + r'$_{f}$' + r'$\/(\frac{energy}{atom})$'
    
    x_stable = [0, 1/3, 0.75, 1]
    y_stable = [0, -0.8, -0.9, 0]
    x_unstable = [0.2]
    y_unstable = [-0.2]
    
    x_imag = [1/3, 1]
    y_imag = [-0.8, 0]
    
    colors = tableau_colors()
    
    m_size = 8
    c_stable, mfc_stable, mec_stable = colors['blue'], 'white', colors['blue']
    m_stable = 'o'
    lw_stable, ls_stable = 1.5, '-'
    ax1 = plt.plot(x_stable, y_stable,
                  color=c_stable,
                  markerfacecolor=mfc_stable,
                  markeredgecolor=mec_stable,
                  marker=m_stable,
                  lw=lw_stable,
                  ls=ls_stable,
                  label='stable',
                  markersize=m_size)

    c_unstable, mfc_unstable, mec_unstable = colors['red'], 'white', colors['red']
    m_unstable = 's'
    lw_unstable, ls_unstable = 0, '-'    
    ax1 = plt.plot(x_unstable, y_unstable,
                  color=c_unstable,
                  markerfacecolor=mfc_unstable,
                  markeredgecolor=mec_unstable,
                  marker=m_unstable,
                  lw=lw_unstable,
                  ls=ls_unstable,
                  label='unstable',
                  markersize=m_size)
    
    ax1 = plt.plot(x_imag, y_imag,
                  color='black',
                  markerfacecolor=mfc_stable,
                  markeredgecolor=mec_stable,
                  marker=m_stable,
                  lw=1,
                  ls='--',
                  label='__nolegend__',
                  markersize=m_size) 
    
    Hd_font = 18
    ax1 = plt.arrow(1/5, -0.8/(1/3)*0.2, 0, abs(-0.2--0.8/(1/3)*0.2)-0.025,
                   length_includes_head=True,
                   head_width=0.01,
                   head_length=0.02,
                   zorder=0,
                   color='black',
                   ls='solid',
                   lw=1)
    ax1 = plt.text(0.21, -0.4, 
                  r'$\Delta$'+r'$\it{H}$'+r'$_{d,A_4B}$',
                  verticalalignment='bottom',
                  color=c_unstable,
                  fontsize=Hd_font)
    ax1 = plt.text(0.74, -0.7, 
                  r'$\Delta$'+r'$\it{H}$'+r'$_{d,AB_3}$',
                  verticalalignment='bottom',
                  horizontalalignment='right',
                  color=c_stable,
                  fontsize=Hd_font)
    
    ax1 = plt.arrow(0.75, -0.8/(-2/3)*0.75-(1/3)*(0.8/(2/3))-0.8-0.005, 0, -0.9-(-0.8/(-2/3)*0.75-(1/3)*(0.8/(2/3))-0.8-0.03),
                   length_includes_head=True,
                   head_width=0.01,
                   head_length=0.02,
                   zorder=0,
                   color='black',
                   ls='solid',
                   lw=1)
    
    ax1 = plt.legend(loc='lower left')
    ax1 = plt.xlabel(xlabel)
    ax1 = plt.ylabel(ylabel)
    ax1 = plt.xticks(xticks)
    ax1 = plt.yticks(yticks)
    ax1 = plt.gca().yaxis.set_ticklabels([])

    ax1 = plt.xlim(xlim)
    ax1 = plt.ylim(ylim)
    
    ax1 = plt.text(0, 0.05, 'A', horizontalalignment='center')
    ax1 = plt.text(1, 0.05, 'B', horizontalalignment='center')    

    ax2 = plt.subplot(122)
    
    d = hullout()
    x, y = [d[c]['Ef'] for c in d], [d[c]['Ed'] for c in d]
    alpha = 0.1
    marker = 'o'
    lw = 0
    s = 10
    stable_c = tableau_colors()['blue']
    unstable_c = tableau_colors()['red']
    colors = [stable_c if y[i] < 0 else unstable_c for i in range(len(y))]
    edgecolors = None
    vmin, vmax, cmap, cmap_values = False, False, False, False
    xticks = (True, [-5, -4, -3, -2, -1, 0, 1, 2])
    yticks = (True, [-1, -0.5, 0, 0.5, 1])
    xlabel = r'$\Delta$' + r'$\it{H}$' + r'$_f$' + r'$\/(\frac{eV}{atom})$'
    ylabel = xlabel.replace('_f', '_d')
    xlim = (-5, 1)
    ylim = (-1, 1)
    diag = False
    ax2 = ax_generic_scatter(x, y, 
                            alpha=alpha,
                            marker=marker,
                            lw=lw,
                            s=s,
                            colors=colors,
                            edgecolors=edgecolors,
                            vmin=vmin,
                            vmax=vmax,
                            cmap=cmap,
                            cmap_values=cmap_values,
                            xticks=xticks,
                            yticks=yticks,
                            xlabel=xlabel,
                            ylabel=ylabel,
                            xlim=xlim,
                            ylim=ylim,
                            diag=diag)
    ax2 = plt.plot(xlim, [0, 0], lw=1, ls='--', color='black')
    
    ax2 = plt.text(-4.9, -0.9, 'stable', color=stable_c)
    ax2 = plt.text(-4.9, 0.9, 'unstable', color=unstable_c, verticalalignment='top')
    m, b, r, p, s = linregress(x, y)
    
    fake_x = np.linspace(xlim[0], xlim[1], 1000)
    fake_y = [m*v+b for v in fake_x]
    ax2 = plt.plot(fake_x, fake_y, lw=1, ls='-', color=tableau_colors()['gray'])
    ax2 = plt.text(-4.95, -0.12, r'$R^2\/=\/%.2f$' % r**2, color=tableau_colors()['gray'], 
                  horizontalalignment='left', verticalalignment='top', fontsize=18)

    ax2 = plt.text(-15, 1.25, 'a', weight='bold')
    ax2 = plt.text(-7, 1.25, 'b', weight='bold')
    ax2 = plt.subplots_adjust(wspace=0.45)
    fig.savefig(os.path.join(FIG_DIR, 'Fig1.png'))
    plt.show()
    plt.close()
    
def make_fig2(experiment):
    training_prop = 'Ef'
    prop = 'Ef'
    compounds = get_compounds(experiment)
    fig = plt.figure(figsize=(9,10))
    ax1 = plt.subplot(321)
    model = 'elfrac'
    actual, pred = get_actual(prop, compounds), get_pred(training_prop, prop, experiment, model, compounds)
    mae = get_mae(actual, pred)
    ax1 = ax_actual_vs_pred(actual, pred, prop,
                            exp=experiment,
                            show_xticks=False, show_yticks=True,
                            show_xlabel=False, show_ylabel=True,
                            show_mae=mae, show_model=model)    
    ax2 = plt.subplot(322)
    model = 'prb14'
    actual, pred = get_actual(prop, compounds), get_pred(training_prop, prop, experiment, model, compounds)
    mae = get_mae(actual, pred)
    ax2 = ax_actual_vs_pred(actual, pred, prop,
                            exp=experiment,
                            show_xticks=False, show_yticks=False,
                            show_xlabel=False, show_ylabel=False,
                            show_mae=mae, show_model=model)    
    ax3 = plt.subplot(323)
    model = 'prb16'
    actual, pred = get_actual(prop, compounds), get_pred(training_prop, prop, experiment, model, compounds)
    mae = get_mae(actual, pred)
    ax3 = ax_actual_vs_pred(actual, pred, prop,
                            exp=experiment,
                            show_xticks=False, show_yticks=True,
                            show_xlabel=False, show_ylabel=True,
                            show_mae=mae, show_model=model)    
    ax4 = plt.subplot(324)
    model = 'npj16'
    actual, pred = get_actual(prop, compounds), get_pred(training_prop, prop, experiment, model, compounds)
    mae = get_mae(actual, pred)
    ax4 = ax_actual_vs_pred(actual, pred, prop,
                            exp=experiment,
                            show_xticks=False, show_yticks=False,
                            show_xlabel=False, show_ylabel=False,
                            show_mae=mae, show_model=model)    
    ax5 = plt.subplot(325)
    model = 'auto'
    actual, pred = get_actual(prop, compounds), get_pred(training_prop, prop, experiment, model, compounds)
    mae = get_mae(actual, pred)
    ax5 = ax_actual_vs_pred(actual, pred, prop,
                            exp=experiment,
                            show_xticks=True, show_yticks=True,
                            show_xlabel=True, show_ylabel=True,
                            show_mae=mae, show_model=model)    
    ax6 = plt.subplot(326)
    model = 'arXiv19'
    actual, pred = get_actual(prop, compounds), get_pred(training_prop, prop, experiment, model, compounds)
    mae = get_mae(actual, pred)
    ax6 = ax_actual_vs_pred(actual, pred, prop,
                            exp=experiment,
                            show_xticks=True, show_yticks=False,
                            show_xlabel=True, show_ylabel=False,
                            show_mae=mae, show_model=model)        

    vmin, vmax = 0, 1
    cticks = (0, 0.2, 0.4, 0.6, 0.8, 1)
        
    add_colorbar(fig, 
                 '|MP - pred|' + r'$\/(\frac{eV}{atom})$', 
                 cticks,
                 'plasma_r',
                 vmin, vmax,
                 [0.95, 0.345, 0.03, 0.3],
                 14, 14, 4, 1.5)
    
    plt.show()
    savename = 'Fig2.png' if experiment == 'allMP' else 'FigS2.png'
    fig.savefig(os.path.join(FIG_DIR, savename))
    plt.close()
    
def make_fig3(training_prop):
    experiment, prop = 'allMP', 'Ed'
    compounds = get_compounds(experiment)
    fig = plt.figure(figsize=(9,10))
    ax1 = plt.subplot(321)
    model = 'elfrac'
    actual, pred = get_actual(prop, compounds), get_pred(training_prop, prop, experiment, model, compounds)
    mae = get_mae(actual, pred)
    ax1 = ax_actual_vs_pred(actual, pred, prop,
                            exp=experiment,
                            show_xticks=False, show_yticks=True,
                            show_xlabel=False, show_ylabel=True,
                            show_mae=mae, show_model=model)
    ax2 = plt.subplot(322)
    model = 'prb14'
    actual, pred = get_actual(prop, compounds), get_pred(training_prop, prop, experiment, model, compounds)
    mae = get_mae(actual, pred)
    ax2 = ax_actual_vs_pred(actual, pred, prop,
                            exp=experiment,
                            show_xticks=False, show_yticks=False,
                            show_xlabel=False, show_ylabel=False,
                            show_mae=mae, show_model=model)    
    ax3 = plt.subplot(323)
    model = 'prb16'
    actual, pred = get_actual(prop, compounds), get_pred(training_prop, prop, experiment, model, compounds)
    mae = get_mae(actual, pred)
    ax3 = ax_actual_vs_pred(actual, pred, prop,
                            exp=experiment,
                            show_xticks=False, show_yticks=True,
                            show_xlabel=False, show_ylabel=True,
                            show_mae=mae, show_model=model)    
    ax4 = plt.subplot(324)
    model = 'npj16'
    actual, pred = get_actual(prop, compounds), get_pred(training_prop, prop, experiment, model, compounds)
    mae = get_mae(actual, pred)
    ax4 = ax_actual_vs_pred(actual, pred, prop,
                            exp=experiment,
                            show_xticks=False, show_yticks=False,
                            show_xlabel=False, show_ylabel=False,
                            show_mae=mae, show_model=model)    
    ax5 = plt.subplot(325)
    model = 'auto'
    actual, pred = get_actual(prop, compounds), get_pred(training_prop, prop, experiment, model, compounds)
    mae = get_mae(actual, pred)
    ax5 = ax_actual_vs_pred(actual, pred, prop,
                            exp=experiment,
                            show_xticks=True, show_yticks=True,
                            show_xlabel=True, show_ylabel=True,
                            show_mae=mae, show_model=model)    
    ax6 = plt.subplot(326)
    model = 'arXiv19'
    actual, pred = get_actual(prop, compounds), get_pred(training_prop, prop, experiment, model, compounds)
    mae = get_mae(actual, pred)
    ax6 = ax_actual_vs_pred(actual, pred, prop,
                            exp=experiment,
                            show_xticks=True, show_yticks=False,
                            show_xlabel=True, show_ylabel=False,
                            show_mae=mae, show_model=model)        

    vmin, vmax = 0, 1
    cticks = (0, 0.2, 0.4, 0.6, 0.8, 1)
        
    add_colorbar(fig, 
                 '|MP - pred|' + r'$\/(\frac{eV}{atom})$', 
                 cticks,
                 'plasma_r',
                 vmin, vmax,
                 [0.95, 0.345, 0.03, 0.3],
                 14, 14, 4, 1.5)
    
    plt.show()
    savename = 'Fig3.png' if training_prop == 'Ef' else 'FigS3.png'
    fig.savefig(os.path.join(FIG_DIR, savename))
    plt.close()
    
def make_fig4(training_prop):
    figsize=(9,10)
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(321)
    ax1 = ax_hist_classification(training_prop, 'elfrac', False, False, 'center left', show_yticks=True)
    ax2 = plt.subplot(322)
    ax2 = ax_hist_classification(training_prop, 'prb14', False, False)
    ax3 = plt.subplot(323)
    ax3 = ax_hist_classification(training_prop, 'prb16', False, True,  show_yticks=True)
    ax4 = plt.subplot(324)
    ax4 = ax_hist_classification(training_prop, 'npj16', False, False)
    ax5 = plt.subplot(325)
    ax6 = ax_hist_classification(training_prop, 'auto', True, False, show_yticks=True)
    ax6 = plt.subplot(326)
    ax6 = ax_hist_classification(training_prop, 'arXiv19', True, False)
    
    plt.show()
    
    savename = 'Fig4.png' if training_prop == 'Ef' else 'FigS4.png'
    fig.savefig(os.path.join(FIG_DIR, savename))
    plt.close()   
    
def make_fig5(training_prop):
    figsize = (9, 10)

    fig = plt.figure(figsize=figsize)
    model = 'elfrac'
    ax1 = plt.subplot(321)
    ax1 = ax_LiMnTMO(training_prop, model, False, True)
    model = 'prb14'
    ax2 = plt.subplot(322)
    ax2 = ax_LiMnTMO(training_prop, model, False, False) 
    model = 'prb16'
    ax3 = plt.subplot(323)
    ax3 = ax_LiMnTMO(training_prop, model, False, True)
    model = 'npj16'
    ax4 = plt.subplot(324)
    ax4 = ax_LiMnTMO(training_prop, model, False, False)
    model = 'auto'
    ax5 = plt.subplot(325)
    ax5 = ax_LiMnTMO(training_prop, model, True, True)
    model = 'arXiv19'
    ax6 = plt.subplot(326)
    ax6 = ax_LiMnTMO(training_prop, model, True, False)    
    
    plt.subplots_adjust(hspace=0.35)
    
    plt.show()
    savename = 'Fig5.png' if training_prop == 'Ef' else 'Fig6.png'
    fig.savefig(os.path.join(FIG_DIR, savename))
    plt.close()   
    
def make_fig6():
    fig = plt.figure(figsize=(16, 3.5))
    ax1 = plt.subplot(131)
    experiment = 'allMP'
    prop = 'Ed'
    training_prop = 'Ed'
    model = 'arXiv19'
    compounds = get_compounds(experiment)
    actual = get_actual(prop, compounds)
    pred = get_pred(training_prop, prop, experiment, model, compounds)
    mae = get_mae(actual, pred)
    ax1 = ax_actual_vs_pred(actual, pred, prop, exp=experiment,
                           show_xticks=True, show_yticks=True,
                           show_xlabel=True, show_ylabel=True,
                           show_mae=mae, show_model=False)
    
    ax2 = plt.subplot(132)
    ax2 = ax_hist_classification(training_prop, model, 
                                 show_xlabel=True, show_ylabel=False, 
                                 leg='upper left', show_model=False, show_yticks=True)
    ax2 = plt.ylabel('No. compounds')
    
    ax3 = plt.subplot(133)
    ax3 = ax_LiMnTMO(training_prop, model, True, True, False)
    
    ax3 = plt.text(-1.65, 0.29, 'a', weight='bold')
    ax3 = plt.text(-0.99, 0.29, 'b', weight='bold')
    ax3 = plt.text(-0.31, 0.29, 'c', weight='bold')

    add_colorbar(fig, 
                 '|MP - pred|' + r'$\/(\frac{eV}{atom})$', 
                 (0, 0.2, 0.4, 0.6, 0.8, 1.),
                 'plasma_r',
                 0, 1,
                 [0.92, 0.0, 0.02, 0.9],
                 14, 14, 4, 1.5)  
    
    """
    experiment = 'smact'
    results = get_results(training_prop, experiment, model)
    compounds, mp_LiMnTMO_stable, pred_LiMnTMO_stable = [results[k] for k in ['compounds', 'MP_stable', 'pred_stable']]
    xpos, ypos = 1, 0.3
    ax3 = plt.text(xpos, ypos, '%i SMACT compounds' % len(compounds))
    ax3 = plt.text(xpos, ypos, '\n%i stable in MP' % len(mp_LiMnTMO_stable))
    ax3 = plt.text(xpos, ypos, '\n\n%i pred. stable' % len(pred_LiMnTMO_stable))
    ax3 = plt.text(xpos, ypos, '\n\n\n%i pred. stable\nand stable in MP' % len(set(mp_LiMnTMO_stable).intersection(pred_LiMnTMO_stable)))
    """
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    fig.savefig(os.path.join(FIG_DIR, 'Fig6.png'))        
    plt.close()
    
def make_fig7():
    fig = plt.figure(figsize=(9.5, 7))
    ax1 = plt.subplot(221)
    experiment = 'allMP'
    prop = 'Ef'
    training_prop = 'Ef'
    model = 'cgcnn'
    compounds = get_compounds(experiment)
    actual = get_actual(prop, compounds)
    pred = get_pred(training_prop, prop, experiment, model, compounds)
    mae = get_mae(actual, pred)
    ax1 = ax_actual_vs_pred(actual, pred, prop, exp=experiment,
                           show_xticks=True, show_yticks=True,
                           show_xlabel=True, show_ylabel=True,
                           show_mae=mae, show_model=False)
    
    ax2 = plt.subplot(222)
    prop = 'Ed'
    actual = get_actual(prop, compounds)
    pred = get_pred(training_prop, prop, experiment, model, compounds)
    mae = get_mae(actual, pred)
    ax2 = ax_actual_vs_pred(actual, pred, prop, exp=experiment,
                           show_xticks=True, show_yticks=True,
                           show_xlabel=True, show_ylabel=True,
                           show_mae=mae, show_model=False)
        
    ax3 = plt.subplot(223)
    ax3 = ax_hist_classification(training_prop, model, 
                                 show_xlabel=True, show_ylabel=False, 
                                 leg='upper left', show_model=False, show_yticks=True)
    ax3 = plt.ylabel('No. compounds')
    
    ax4 = plt.subplot(224)
    ax4 = ax_LiMnTMO(training_prop, model, True, True, False)
    
    ax4 = plt.text(-1, 0.82, 'a', weight='bold')
    ax4 = plt.text(-0.3, 0.82, 'b', weight='bold')
    ax4 = plt.text(-1, 0.3, 'c', weight='bold')
    ax4 = plt.text(-0.3, 0.3, 'd', weight='bold')

    
    add_colorbar(fig, 
                 '|MP - pred|' + r'$\/(\frac{eV}{atom})$', 
                 (0, 0.2, 0.4, 0.6, 0.8, 1.),
                 'plasma_r',
                 0, 1,
                 [0.95, 0.3, 0.025, 0.35],
                 14, 14, 4, 1.5)
    
    plt.subplots_adjust(hspace=0.55, wspace=0.5)
    plt.show()
    
    fig.savefig(os.path.join(FIG_DIR, 'Fig7.png'))        
    plt.close()
    
def make_figS1():
    compounds = get_compounds('allMP')
    x, y = get_actual('Ef', compounds), get_actual('Ed', compounds)
    fig = plt.figure()
    ax = plt.subplot(111)
    nbins = 85
    alpha=0.5
    norm = True
    xlim = (-4, 1)
    label_Hf = r'$\Delta$'+r'$\it{H}$'+r'$_{f,MP}$'
    label_Hd = label_Hf.replace('f', 'd')
    ax = plt.hist(x, 
                  color=tableau_colors()['purple'], 
                  bins=nbins,
                  alpha=alpha,
                  density=norm,
                  label=label_Hf)
    ax = plt.hist(y, 
                  color=tableau_colors()['orange'], 
                  bins=nbins,
                  alpha=alpha,
                  density=norm,
                  label=label_Hd)
    
    ax = plt.xlim(xlim)
    
    ax = plt.gca().yaxis.set_ticklabels([])
    ax = plt.ylabel('Norm. frequency')    
    ax = plt.xlabel(label_Hf+' or '+label_Hd+r'$\/(\frac{eV}{atom})$')     
    ax = plt.xticks((-4, -3, -2, -1, 0, 1))
    #ax = plt.legend()
    ax = plt.text(-2, 0.5, 
                  label_Hf, 
                  color=tableau_colors()['purple'],
                  horizontalalignment='center')
    ax = plt.text(-0.6, 2, 
                  label_Hd, 
                  color=tableau_colors()['orange'],
                  horizontalalignment='center')
    plt.show()
    fig.savefig(os.path.join(FIG_DIR, 'FigS1.png'))
    plt.close()    
    
def make_table1(training_prop):
    
    x = PrettyTable()
    
    models = ['elfrac', 'prb14', 'prb16', 'npj16', 'auto', 'arXiv19']
    
    x.field_names = ['']+models
    
    row_names = ['candidate compositions',
                 'stable compounds in MP',
                 'compounds pred. stable',
                 '% predicted stable',
                 'pred. stable and MP stable']    
    
    experiment = 'smact'
    data = {model : get_results(training_prop, experiment, model) for model in models}  
    
    row1 = [row_names[0]]+[len(data[m]['compounds']) for m in models]
    row2 = [row_names[1]]+[len(data[m]['MP_stable']) for m in models]
    row3 = [row_names[2]]+[len(data[m]['pred_stable']) for m in models]
    row4 = [row_names[3]]+[np.round(100*len(data[m]['pred_stable'])/len(data[m]['compounds']), 1) for m in models]
    row5 = [row_names[4]]+[len(set(data[m]['MP_stable']).intersection(set(data[m]['pred_stable']))) for m in models]
    
    rows = [row1, row2, row3, row4, row5]
    for r in rows:
        x.add_row(r)
    
    name = 'Table1' if training_prop == 'Ef' else 'TableS1'
    with open(os.path.join(FIG_DIR, name), 'w') as f:
        f.write(str(x))
    
def main():
    set_rc_params()    
    remake_fig1 = True
    remake_fig2 = True
    remake_fig3 = True
    remake_fig4 = True
    remake_fig5 = True
    remake_table1 = True
    remake_fig6 = True
    remake_fig7 = True
    remake_figS1 = True
    remake_figS2 = True
    remake_figS3 = True
    remake_figS4 = True
    remake_tableS1 = True
    
    if remake_fig1:
        make_fig1()
    if remake_fig2:
        make_fig2('allMP')
    if remake_fig3:
        make_fig3('Ef')
    if remake_fig4:
        make_fig4('Ef')
    if remake_fig5:
        make_fig5('Ef')
    if remake_fig6:
        make_fig5('Ed')
    if remake_fig7:
        make_fig7()
    if remake_figS1:
        make_figS1()
    if remake_figS2:
        make_fig2('LiMnTMO')
    if remake_figS3:
        make_fig3('Ed')
        print('!!!!!!\nroost data placeholder for auto\n!!!!!')
    if remake_figS4:
        make_fig4('Ed')
        print('!!!!!!\nroost data placeholder for auto\n!!!!!')
    if remake_table1:
        make_table1('Ef')
    if remake_tableS1:
        make_table1('Ed')
    return

if __name__ == '__main__':
    main()