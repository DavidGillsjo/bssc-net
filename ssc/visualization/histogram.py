import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import torch
import os.path as osp

def getFigure(resolution=[800, 600]):
    dpi = 200.0
    resolution = np.array(resolution, dtype=np.float)
    return plt.figure(figsize=resolution/dpi, dpi = dpi)

class HistogramGroup:
    '''
    Metrics and limits are shared by all histograms
    Data can be partitioned differently, this done by adding types.
    metrics: Dict of metric:limits
    '''
    def __init__(self, metrics, nbr_bins=10, metric_pairs = [], make_pdf = False):
        self.metrics = metrics
        self.metric_pairs = metric_pairs
        self.histograms = {}
        self.nbr_bins = nbr_bins
        self.make_pdf = make_pdf
        self.histograms_2d = {}

    def add_type(self, name, classes):
        self.histograms[name] = {}
        for metric, limits in self.metrics.items():
            self.histograms[name][metric] = HistogramTpFn(classes, limits, self.nbr_bins)

        self.histograms_2d[name] = {}
        for xy_metric in self.metric_pairs:
            limits = [self.metrics[m] for m in xy_metric]
            self.histograms_2d[name][xy_metric] = HistogramTpFn2D(classes, limits, 2*[self.nbr_bins], metric = xy_metric)


    def add(self, pred, target, metric_dict, masks_dict = {}):
        '''
        Assumes that metric_dict shares keys with self.metrics
        Assumes that masks_dict shares keys with self.histograms
        '''
        for metric in self.metrics:
            if metric not in metric_dict:
                continue
            values = metric_dict[metric]
            for type, histograms in self.histograms.items():
                mask = masks_dict.get(type, None)
                histograms[metric].add(pred, target, values, mask)

        for xy_metric in self.metric_pairs:
            if not set(xy_metric).issubset(metric_dict.keys()):
                continue
            x_values = metric_dict[xy_metric[0]]
            y_values = metric_dict[xy_metric[1]]
            for type, histograms in self.histograms_2d.items():
                mask = masks_dict.get(type, None)
                histograms[xy_metric].add(pred, target, x_values, y_values, mask)

    def reset(self):
        for type, histograms in self.histograms.items():
            for metric, hist in histograms.items():
                hist.reset()
        for type, histograms in self.histograms_2d.items():
            for xy_metric, hist in histograms.items():
                hist.reset()

    def plot_tb(self, tblogger, epoch):
        for type, histograms in self.histograms.items():
            for metric, hist in histograms.items():
                hist_fig = hist.plot()
                if self.make_pdf:
                    fname = 'hist_{}_{}_e{:03d}'.format(metric, type, epoch)
                    hist_fig.savefig(osp.join(tblogger.log_dir, fname))
                tblogger.add_figure('hist_{}_{}'.format(metric, type), hist_fig, global_step = epoch, close = True)

        for type, histograms in self.histograms_2d.items():
            for xy_metric, hist in histograms.items():
                hist_fig = hist.plot()
                if self.make_pdf:
                    fname = 'hist_{}_{}_{}_e{:03d}'.format(*xy_metric, type, epoch)
                    hist_fig.savefig(osp.join(tblogger.log_dir, fname))
                tblogger.add_figure('hist_{}_{}_{}'.format(*xy_metric, type), hist_fig, global_step = epoch, close = True)


    def get_classes(self, type):
        for _, hist in self.histograms[type].items():
            return hist.classes

    def plot_bhattacharyya_tb(self, tblogger, epoch):
        # Calculate for first type of histogram, will be the same for all types
        for type, histograms in self.histograms.items():
            for metric, hist in histograms.items():
                bh_coef = hist.bhattacharyya_coef()
                tblogger.add_scalar('bhattacharyya_{}_{}'.format(metric, type), bh_coef, global_step = epoch)

        for type, histograms in self.histograms_2d.items():
            for xy_metric, hist in histograms.items():
                bh_coef = hist.bhattacharyya_coef()
                tblogger.add_scalar('bhattacharyya_{}_{}_{}'.format(*xy_metric, type), bh_coef, global_step = epoch)


class HistogramTpFn:
    '''
    Accumulates statistics in histogram form based on True Positive or False Negative.
    Classes are separated by class_masks, if not given the classes are simply the classes given by target and pred.
    Tries to use GPU if possible by utilizings torch.histc function instead of np.histogram.
    '''
    def __init__(self, classes = range(10), limits = [0, 1], nbr_bins = 10):
        self.nbr_bins = nbr_bins
        self.limits = limits
        self.classes = classes
        self.nbr_classes = len(classes)
        bin_edges = np.linspace(limits[0], limits[1], nbr_bins+1)
        self.left_edges = bin_edges[:-1]
        self.bin_width = bin_edges[1] - bin_edges[0]
        self.histograms = {}
        for l in ['tp', 'fn']:
            self.histograms[l] = np.zeros([self.nbr_classes, nbr_bins], dtype=np.int64)

    def add(self, pred, target, metric, class_masks = None):
        '''
        class_masks = [Batch x Masks x DataDim]
        '''
        tp = (pred==target)
        for pl, p_mask in zip(['tp', 'fn'], [tp, ~tp]):
            for cl in range(self.nbr_classes):
                class_mask = (target == cl) if class_masks is None else class_masks[:, cl]
                mask = p_mask & class_mask
                if mask.any():
                    hist = torch.histc(metric[mask], self.nbr_bins, *self.limits)
                    self.histograms[pl][cl] += hist.to('cpu').type(torch.int64).numpy()

    def reset(self):
        for l in ['tp', 'fn']:
            for i in range(self.nbr_classes):
                self.histograms[l][i].fill(0)

    def plot(self):
        fg = plt.figure()
        nbr_rows = np.ceil(np.sqrt(10.0*self.nbr_classes/16.0))
        nbr_cols = np.ceil(self.nbr_classes/nbr_rows)
        for ci, cl in enumerate(self.classes):
            plt.subplot(nbr_rows, nbr_cols, ci+1)
            plt.title(cl)
            plt.bar(self.left_edges, self.histograms['tp'][ci], self.bin_width,
                    align = 'edge')
            plt.bar(self.left_edges, self.histograms['fn'][ci], self.bin_width,
                    bottom = self.histograms['tp'][ci], align = 'edge')
        plt.legend(['tp', 'fn'])
        plt.tight_layout()

        return fg

    def bhattacharyya_coef(self):
        '''
        Measure distance between the TP and FN distributions.
        Normalize with total count.
        Compute joint coef for all classes.
        '''
        tp_count = np.sum(self.histograms['tp'], axis=0)
        fn_count = np.sum(self.histograms['fn'], axis=0)
        count = np.sum(tp_count) + np.sum(fn_count)

        bhattacharyya = np.sum(np.sqrt(tp_count*fn_count))/count if count > 0 else np.NaN
        return bhattacharyya


class HistogramTpFn2D(HistogramTpFn):
    def __init__(self, classes = range(10), limits = ([0, 1], [0, 1]), nbr_bins = (10, 10), metric=[None, None]):
        self.metric = metric
        self.nbr_bins = nbr_bins
        self.limits = limits
        self.classes = classes
        self.nbr_classes = len(classes)
        self.bin_edges = [np.linspace(l[0], l[1], nb+1) for l,nb in zip(limits, nbr_bins)]
        self.histograms = {}
        for l in ['tp', 'fn']:
            self.histograms[l] = np.zeros([self.nbr_classes, *nbr_bins], dtype=np.int64)

    def add(self, pred, target, metric1, metric2, class_masks = None):
        '''
        class_masks = [Batch x Masks x DataDim]
        '''
        tp = (pred==target)
        for pl, p_mask in zip(['tp', 'fn'], [tp, ~tp]):
            for cl in range(self.nbr_classes):
                class_mask = (target == cl) if class_masks is None else class_masks[:, cl]
                mask = p_mask & class_mask
                if mask.any():
                    hist, _, _ = np.histogram2d(metric1[mask].to('cpu').numpy(), metric2[mask].to('cpu').numpy(), self.nbr_bins, range=self.limits)
                    self.histograms[pl][cl] += hist.astype(np.int64)

    def plot(self):
        fg = plt.figure()
        # fg.suptitle('Recall [tp/(tp + fn)]')
        fg.suptitle('Square: #Total, Circle: Recall')
        nbr_rows = np.ceil(np.sqrt(10.0*self.nbr_classes/16.0))
        nbr_cols = np.ceil(self.nbr_classes/nbr_rows)
        for ci, cl in enumerate(self.classes):
            ax = plt.subplot(nbr_rows, nbr_cols, ci+1)
            plt.title(cl)
            X, Y = np.meshgrid(*self.bin_edges)
            tp = self.histograms['tp'][ci].T
            fn = self.histograms['fn'][ci].T
            with np.errstate(invalid='ignore'):
                recall = tp/(tp+fn)
            # max_count = np.amax(tp+fn)
            # with np.errstate(invalid='ignore'):
            #     ax.pcolormesh(X, Y, np.transpose(tp / (tp + fn)))
            ax.pcolormesh(X, Y, tp+fn, vmin = 0)
            bin_centers = [(be[1:] + be[:-1])/2 for be in self.bin_edges]
            X_center, Y_center = np.meshgrid(*bin_centers)
            ax.scatter(X_center.ravel(), Y_center.ravel(), c = recall.ravel(), vmin = 0, vmax = 1.0, s = 9)
            if self.metric[1] and ci % nbr_cols == 0:
                plt.ylabel(self.metric[1])
            if self.metric[0] and ci >= self.nbr_classes - nbr_cols:
                plt.xlabel(self.metric[0])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fg
