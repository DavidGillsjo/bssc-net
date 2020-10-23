import numpy as np
import matplotlib.pyplot as plt
import torch
import os.path as osp

def plot_sample_net(mod_name, conv_mod, net):
    weights = conv_mod.weight.data.to('cpu').numpy().squeeze()
    if conv_mod.bias:
        bias = conv_mod.bias.data.to('cpu').numpy().squeeze()
        nbr_bias = bias.size
    else:
        nbr_bias = 0
    metrics = net.metrics
    nbr_metrics = len(metrics)
    nbr_classes = weights.shape[0]

    if nbr_metrics == weights.size:
        fig = plt.figure()
        if nbr_bias > 0:
            nbr_bars = nbr_metrics+nbr_bias
            bar_data = np.append(weights, bias)
            bar_labels = metrics+['bias']
        else:
            nbr_bars = nbr_metrics
            bar_data = weights
            bar_labels = metrics
        plt.bar(range(nbr_bars), bar_data, tick_label=bar_labels)
        return fig


    try:
        weights = weights.reshape([nbr_classes, nbr_metrics, nbr_classes])
    except ValueError:
        return None

    fig = plt.figure()
    for i,m in enumerate(metrics):
        plt.subplot(1,nbr_metrics,i+1)
        plt.imshow(weights[:,i,:])
        plt.title(m)
        plt.colorbar()

    return fig
