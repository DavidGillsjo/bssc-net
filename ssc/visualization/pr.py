import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from ssc.utils.metrics import compute_confusion
import math
import operator
import os.path as osp

def count_confusion(cm):
    all_positive = torch.sum(cm, dim=0)
    tp = torch.diag(cm)
    all_gt = torch.sum(cm, dim=1)

    return {'tp': tp, 'pos': all_positive, 'gt': all_gt}

class PRGroup:
    '''
    Data can be partitioned differently, this done by adding types.
    metrics: List of metric names
    limits, list of limits per metric
    step_size: list of step size per metric
    '''
    def __init__(self, metrics, limits = [], step_size = [], direct_prop = [], device = 'cpu', make_pdf = False):
        self.metrics = metrics
        self.limits = limits if limits else len(metrics)*[0.05, 0.95]
        self.step_size = step_size if step_size else len(metrics)*[0.05]
        self.device = device
        self.direct_prop = direct_prop
        self.make_pdf = make_pdf
        self.pr = {}

    def add_type(self, name, classes, mask_classes = None):
        self.pr[name] = {}
        for metric, limits, step_size, direct_prop in zip(self.metrics, self.limits, self.step_size, self.direct_prop):
            self.pr[name][metric] = PR(classes, limits, step_size, mask_classes = mask_classes, direct_prop = direct_prop, device = self.device)

    def add(self, pred, target, metric_dict, masks_dict = {}):
        '''
        Assumes that metric_dict shares keys with self.metrics
        Assumes that masks_dict shares keys with self.pr
        '''
        for metric in self.metrics:
            if metric not in metric_dict:
                continue
            values = metric_dict[metric]
            for type, pr in self.pr.items():
                mask = masks_dict.get(type, None)
                pr[metric].add(pred, target, values, mask)

    def reset(self):
        for type, pr_type in self.pr.items():
            for metric, pr in pr_type.items():
                pr.reset()

    def _tb_table(self, tblogger, epoch):
        h_sem_classes = None
        table = ''
        for type, pr_type in self.pr.items():
            for metric, pr in pr_type.items():
                #Check if new header is needed
                if np.any(pr.sem_classes != h_sem_classes):
                    table += '|Type|Mask|Metric|mAP|wmAP|AP:|' + '|'.join(pr.sem_classes) + '|\n'
                    table += '|' + '---|'*(6+pr.nbr_sem_classes) + '\n'
                    h_sem_classes = pr.sem_classes
                for mci, mc in enumerate(pr.mask_classes):
                    table += '|{}|{}|{}|{:.2f}|{:.2f}| |'.format(type, mc, metric, pr.PR['mAP'][mci], pr.PR['wmAP'][mci])
                    for ap in pr.PR['AP'][mci]:
                        table += '{:.2f}|'.format(ap)
                    table += '\n'

        tblogger.add_text('Precision Recall Stats', table, global_step=epoch)

    def _tb_mAP(self, tblogger, epoch):
        for type, pr_type in self.pr.items():
            for metric, pr in pr_type.items():
                for ap in ['mAP', 'wmAP']:
                    if pr.nbr_mask_classes == 1:
                        tblogger.add_scalar('{}_{}_{}'.format(ap, metric, type), pr.PR[ap][0], global_step = epoch)

    def _tb_mIoU_AuC(self, tblogger, epoch):
        for type, pr_type in self.pr.items():
            for metric, pr in pr_type.items():
                if pr.nbr_mask_classes == 1:
                    tblogger.add_scalar('{}_{}_{}'.format('mIoU_AuC', metric, type), pr.IoU['mIoU_AuC'][0], global_step = epoch)


    def plot_tb(self, tblogger, epoch):
        for type, pr_type in self.pr.items():
            for metric, pr in pr_type.items():
                pr.compute_PR()
                PR_fig = pr.plot_PR()
                ROC_fig = pr.plot_ROC()
                if self.make_pdf:
                    fname = 'PR_{}_{}_{:03d}'.format(metric, type, epoch)
                    PR_fig.savefig(osp.join(tblogger.log_dir, fname))
                    fname = 'ROC_{}_{}_{:03d}'.format(metric, type, epoch)
                    ROC_fig.savefig(osp.join(tblogger.log_dir, fname))
                tblogger.add_figure('PR_{}_{}'.format(metric, type), PR_fig, global_step = epoch, close = True)
                tblogger.add_figure('ROC_{}_{}'.format(metric, type), ROC_fig, global_step = epoch, close = True)

        self._tb_table(tblogger, epoch)
        self._tb_mAP(tblogger, epoch)
        self._tb_mIoU_AuC(tblogger, epoch)

    def get_sem_classes(self):
        for _, pr in self.pr[type].items():
            return pr.sem_classes


class PR:
    '''
    Accumulates statistics in TP,FP,TN,FN form for later calculation of Precision Recall metrics.
    Classes are separated by class_masks, if not given the classes are simply the classes given by target and pred.
    Tries to use GPU if possible.
    PASCAL uses step size 0.1
    COCO uses step size 0.01
    '''
    def __init__(self, classes = range(10), limits = [0.0, 1.0], step_size = 0.01, mask_classes = None, direct_prop = True, device = 'cpu'):
        self.limits = limits
        self.sem_classes = classes
        self.nbr_sem_classes = len(classes)
        self.mask_classes = mask_classes if mask_classes else ['semantic']
        self.nbr_mask_classes = len(self.mask_classes)
        self.thresholds = torch.linspace(limits[1], limits[0], math.ceil((limits[1]-limits[0])/step_size) + 1)
        self.device = device
        self.comparison = operator.gt if direct_prop else operator.lt


        self.labels = ['tp', 'pos', 'gt']
        self.counts = {}
        for l in self.labels:
            dims = [self.nbr_mask_classes, self.nbr_sem_classes]
            if l == 'gt':
                dims += [1]
            else:
                dims += [self.thresholds.numel()]
            self.counts[l] = torch.zeros(dims, dtype=torch.long, device = device)


    def add(self, pred, target, metric, class_masks = None):
        '''
        class_masks = [Batch x Masks x DataDim]
        '''
        for cl in range(self.nbr_mask_classes):

            if class_masks is None:
                class_pred = pred
                class_target = target
                class_metric =  metric
            else:
                class_pred = pred[class_masks[:, cl]]
                class_target = target[class_masks[:, cl]]
                class_metric =  metric[class_masks[:, cl]]

            cm = torch.zeros([self.nbr_sem_classes, self.nbr_sem_classes], dtype = torch.long, device=self.device)
            gt = torch.zeros(self.nbr_sem_classes, dtype = torch.long, device=self.device)
            gt_idx, gt_count = torch.unique(target, return_counts=True)
            gt[gt_idx] = gt_count
            self.counts['gt'][cl,:,0] += gt

            for ti, t in enumerate(self.thresholds):
                t_mask = self.comparison(class_metric, t)
                cm += compute_confusion(class_pred[t_mask], class_target[t_mask], self.nbr_sem_classes)
                self.counts['tp'][cl,:,ti] += torch.diag(cm)
                self.counts['pos'][cl,:,ti] += torch.sum(cm, dim=0)
                class_pred = class_pred[~t_mask]
                class_target = class_target[~t_mask]
                class_metric = class_metric[~t_mask]


    def reset(self):
        for l in self.labels:
            self.counts[l].fill_(0)


    def compute_PR(self):
        tp = self.counts['tp'].to(dtype=torch.float)
        pos = self.counts['pos'].to(dtype=torch.float)
        gt = self.counts['gt'].to(dtype=torch.float)
        zero_mat = torch.zeros_like(tp, device=self.device)
        one_mat = torch.ones_like(tp, device=self.device)
        PR = {}
        PR['P'] = P = torch.where(pos!=0, tp/pos, one_mat)
        PR['mP'] = mP = torch.mean(P, dim=1)
        all_pos = pos.sum(dim=1, keepdim=True)
        ratio_P = torch.where(all_pos!=0, pos/all_pos, one_mat)
        PR['wmP'] = wmP = torch.sum(ratio_P*P, dim=1)
        PR['R'] = R = torch.where(gt!=0, tp/gt, zero_mat)
        PR['mR'] = mR = torch.mean(R, dim=1)
        all_gt = gt.sum(dim=1, keepdim = True)
        ratio_R = gt/all_gt
        PR['wmR'] = wmR = torch.sum(ratio_R*R, dim=1)

        # interp_P = P
        # interp_mP = mP
        # interp_wmP = wmP

        interp_P = torch.zeros_like(P, device = self.device)
        interp_mP = torch.zeros_like(mP, device = self.device)
        interp_wmP = torch.zeros_like(wmP, device = self.device)
        for ti, t in enumerate(self.thresholds):
            interp_P[:,:,ti], _ = torch.max(P[:,:,ti:], dim=2)
            interp_mP[:,ti], _ = torch.max(mP[:,ti:], dim=1)
            interp_wmP[:,ti], _ = torch.max(wmP[:,ti:], dim=1)

        left_R = torch.roll(R, 1, dims=2)
        left_R[:,:,0] = 0
        PR['AP'] = torch.sum(interp_P*(R - left_R), dim=2)
        left_mR = torch.roll(mR, 1, dims=1)
        left_mR[:,0] = 0
        PR['mAP'] = torch.sum(interp_mP*(mR - left_mR), dim=1)
        left_wmR = torch.roll(wmR, 1, dims=1)
        left_wmR[:,0] = 0
        PR['wmAP'] = torch.sum(interp_wmP*(wmR - left_wmR), dim=1)

        #Move to CPU and numpy for plotting
        self.PR = {}
        for m, v in PR.items():
            self.PR[m] = v.to('cpu').numpy()

        ROC = {}
        ROC['tpr'] = P
        fp = pos - tp
        neg = all_pos - pos
        ROC['fpr'] = fp/neg
        self.ROC = {}
        for m, v in ROC.items():
            self.ROC[m] = v.to('cpu').numpy()

        IoU = {}
        intersection = tp
        union = pos + gt
        IoU['IoU'] = torch.where(union!=0, intersection/union, zero_mat)
        IoU['mIoU'] = mIoU = torch.mean(IoU['IoU'], dim=1)
        interp_mIoU = torch.zeros_like(mIoU, device = self.device)
        for ti, t in enumerate(self.thresholds):
            interp_mIoU[:,ti], _ = torch.max(mIoU[:,ti:], dim=1)
        IoU['mIoU_AuC'] = torch.sum(interp_mIoU*(mR - left_mR), dim=1)

        self.IoU = {}
        for m, v in IoU.items():
            self.IoU[m] = v.to('cpu').numpy()



    def plot_PR(self):
        fg = plt.figure()
        nbr_rows = np.ceil(np.sqrt(10.0*self.nbr_mask_classes/16.0))
        nbr_cols = np.ceil(self.nbr_mask_classes/nbr_rows)

        colors = sns.color_palette("husl", self.nbr_sem_classes)

        for mci, mc in enumerate(self.mask_classes):
            plt.subplot(nbr_rows, nbr_cols, mci+1)
            for (sci, scl) in enumerate(self.sem_classes):
                plt.plot(self.PR['R'][mci, sci], self.PR['P'][mci, sci], color=colors[sci], label = '{}, AP: {:.2f}'.format(scl, self.PR['AP'][mci,sci]))

            plt.title(self.mask_classes[mci])
            plt.ylim([min(0.5, np.min(self.PR['P'])), 1])
            plt.xlim([min(0.5, np.min(self.PR['R'])), 1])
            plt.legend(loc='lower left', ncol = 2)

            if mci % nbr_cols == 0:
                plt.ylabel('Precision')

            if nbr_rows == 1 or mci > self.nbr_mask_classes - nbr_cols:
                plt.xlabel('Recall')

        plt.tight_layout()

        return fg

    def plot_ROC(self):
        fg = plt.figure()
        nbr_rows = np.ceil(np.sqrt(10.0*self.nbr_mask_classes/16.0))
        nbr_cols = np.ceil(self.nbr_mask_classes/nbr_rows)

        colors = sns.color_palette("husl", self.nbr_sem_classes)

        for mci, mc in enumerate(self.mask_classes):
            plt.subplot(nbr_rows, nbr_cols, mci+1)
            for (sci, scl) in enumerate(self.sem_classes):
                plt.plot(self.ROC['fpr'][mci, sci], self.ROC['tpr'][mci, sci], color=colors[sci], label = scl)

            plt.title(self.mask_classes[mci])
            plt.legend(loc='lower right', ncol = 2)

            if mci % nbr_cols == 0:
                plt.ylabel('True Positive Rate')

            if nbr_rows == 1 or mci > self.nbr_mask_classes - nbr_cols:
                plt.xlabel('False Positive Rate')

        plt.tight_layout()

        return fg
