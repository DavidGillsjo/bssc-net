import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import argparse
import os.path as osp
from torch.utils.data import DataLoader
from datetime import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import git
from importlib import import_module
from ssc.net.bayesian.layers import BNLLLoss, BConv
from matplotlib import pyplot as plt

from ssc.utils.metrics import compute_confusion, plot_confusion, count_parameters

from ssc.scripts.train import TrainNet, seed
from torchvision import datasets, transforms
from ssc.visualization.histogram import HistogramGroup
from ssc.visualization.pr import PRGroup, count_confusion
from ssc.visualization.weight import plot_sample_net


class TrainOnMNIST(TrainNet):

    def init_histograms(self):
        self.histograms = HistogramGroup({
                                        'avg_max_scores': [0, 1.0],
                                        'entropy_top': [0, 0.3],
                                        'var_e_top': [0, 0.2],
                                        'var_a_top': [0, 0.2]
                                        },
                                         metric_pairs = [
                                             ('avg_max_scores', 'entropy_top')
                                         ],
                                         make_pdf =self.cfg.get('make_pdf', False))
        self.histograms.add_type('semantic', self.val_dataloader.dataset.classes)

    def init_prs(self):
        self.prs = PRGroup(['avg_max_scores', 'entropy_top'],
                          [[0.5, 0.95], [0.05, 0.5]],
                          [0.05, 0.05], [True, False], device=self.device,
                          make_pdf =self.cfg.get('make_pdf', False))

        self.prs.add_type('semantic', self.val_dataloader.dataset.classes)

    def run_epoch(self):
        train = self.net.training
        dataloader = self.train_dataloader if train else self.val_dataloader
        tblogger = self.tblogger['train'] if train else self.tblogger['val']
        acc_loss = 0
        acc_kl = 0
        if not train:
            confusion = torch.zeros([self.nclass, self.nclass], dtype=torch.long, device=self.device)
            acc_score = 0
            self.init_histograms()
            self.init_prs()

        for i_batch, (data, target) in enumerate(dataloader):

            batch_size = data.shape[0]
            first_val = (i_batch == 0) and not train
            sample_output = (not train)

            if train and self.cfg.get('excluded_labels', []):
                l_mask = torch.ones(batch_size, dtype=torch.bool)
                for l in self.cfg['excluded_labels']:
                    l_mask &= (target != l)
                target = target[l_mask]
                data = data[l_mask]
                batch_size = data.shape[0]

            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            result = self.forward(data, target, i_batch)
            loss = result['loss']
            kl = result['kl']
            log_score = result['log_score']

            # Zero gradients, perform a backward pass, and update the weights.
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Metrics
            batch_nbr = self.epoch*len(dataloader) + i_batch
            acc_loss += loss.item()
            acc_kl += kl
            tblogger.add_scalar('batch_loss', loss.item(), global_step = batch_nbr)
            tblogger.add_scalar('batch_kl', kl, global_step = batch_nbr)

            if not train:
                if 'mean_score' in result:
                    try:
                        score, pred_batch = result['fused_score'].max(1)
                    except KeyError:
                        score, pred_batch = result['mean_score'].max(1)
                    b_idx = torch.arange(batch_size, dtype=int)
                    result['avg_max_scores'] = score
                    result['var_a_top'] = result['var_aleatoric'][b_idx, pred_batch]
                    result['var_e_top'] = result['var_epistemic'][b_idx, pred_batch]
                    result['var_top'] = result['var_a_top'] + result['var_e_top']
                    result['entropy_top'] = result['entropy'][b_idx, pred_batch]

                else:
                    score, pred_batch = log_score.max(1)
                    score = score.exp()
                    result['avg_max_scores'] = score

                self.histograms.add(pred_batch, target, result)
                self.prs.add(pred_batch, target, result)

                batch_confusion = compute_confusion(pred_batch, target, self.nclass)
                confusion += batch_confusion
                acc_score += score.sum()

            # Plot graph on first epoch and batch
            # if self.first_epoch and i_batch == 0:
            #     if self.bayesian and not self.net.fixed:
            #         self.net.fix_parameters()
            #         tblogger.add_graph(self.net.bnn, data)
            #         self.net.release_parameters()
            #     else:
            #         tblogger.add_graph(self.net, data)

            # Plot examples in first validation batch
            if first_val:
                for ex_idx in range(min(self.cfg['nbr_examples'], batch_size)):
                    ex_fig = plt.figure()
                    plt.imshow(data[ex_idx].cpu().numpy().squeeze())
                    title_str = 'Pred: {}, GT: {}'.format(
                        dataloader.dataset.classes[pred_batch[ex_idx]],
                        dataloader.dataset.classes[target[ex_idx]]
                    )
                    if 'mean_score' in result:
                        title_str += '\nscore: {:.2g}, var_a: {:.2g}, var_e: {:.2g}'.format(
                            score[ex_idx],
                            result['var_a_top'][ex_idx],
                            result['var_e_top'][ex_idx]
                        )
                    else:
                        title_str += '\nscore: {:.2g}'.format(score[ex_idx])
                    plt.title(title_str)

                    tblogger.add_figure("Example {}".format(ex_idx), ex_fig, global_step = self.epoch, close = True)

                if 'mean_score' in result:
                    corr_mask = (pred_batch == target).to('cpu').numpy()
                    var_fig = plt.figure()
                    r_idx = numpy.arange(batch_size)
                    for ki, key in enumerate(['avg_max_scores', 'var_a_top', 'var_e_top', 'var_top', 'entropy_top']):
                        r = result[key].to('cpu').numpy()
                        plt_f = plt.semilogy if 'var' in key else plt.plot
                        plt.subplot(2,3,ki+1)
                        plt_f(r_idx[corr_mask],r[corr_mask].flat, '.', label='Correct',)
                        plt_f(r_idx[~corr_mask], r[~corr_mask].flat, '.', label='Incorrect')
                        plt.title(key)
                        if ki == 1:
                            plt.legend()
                    plt.tight_layout()
                    tblogger.add_figure("Variance", var_fig, global_step = self.epoch, close = True)


        #Accumulate metrics
        tblogger.add_scalar('epoch_loss', acc_loss/len(dataloader.dataset), global_step = self.epoch)
        tblogger.add_scalar('epoch_kl', acc_kl/len(dataloader.dataset), global_step = self.epoch)
        tblogger.add_scalar('lr', self.lr, global_step = self.epoch)
        if train:
            all_weights = []
            all_weights_mu = []
            all_weights_sigma = []
            for m_name, m in self.net.named_modules():
                if isinstance(m, BConv):
                    sigma = torch.exp(m.sigma_weight.data)
                    tblogger.add_histogram(m_name + '_mu', m.mu_weight.data, self.epoch)
                    tblogger.add_histogram(m_name + '_sigma', sigma, self.epoch)
                    all_weights.append(m.sample_weights().data.reshape(-1))
                    all_weights_mu.append(m.mu_weight.data.reshape(-1))
                    all_weights_sigma.append(sigma.reshape(-1))
                elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    tblogger.add_histogram(m_name, m.weight.data, self.epoch)
                    all_weights.append(m.weight.data.reshape(-1))

                    if 'sample_net' in m_name:
                        sample_w_fig = plot_sample_net(m_name, m, self.net)
                        if sample_w_fig:
                            tblogger.add_figure("Sample weights{}".format(m_name), sample_w_fig, global_step = self.epoch, close = True)

            tblogger.add_histogram('all_weights', torch.cat(all_weights), self.epoch)
            if all_weights_mu:
                tblogger.add_histogram('all_weights_mu', torch.cat(all_weights_mu), self.epoch)
                tblogger.add_histogram('all_weights_sigma', torch.cat(all_weights_sigma), self.epoch)

        else:
            accuracy = 100.0 * float(torch.trace(confusion))/len(dataloader.dataset)
            tblogger.add_scalar('accuracy', accuracy, global_step = self.epoch)
            tblogger.add_scalar('avg_score', acc_score/len(dataloader.dataset), global_step = self.epoch)
            conf_fig = plot_confusion(confusion.to('cpu').numpy(), dataloader.dataset.classes)
            tblogger.add_figure("Confusion Matrix", conf_fig, global_step = self.epoch, close = True)
            self.histograms.plot_tb(tblogger, self.epoch)
            self.histograms.plot_bhattacharyya_tb(tblogger, self.epoch)
            self.prs.plot_tb(tblogger, self.epoch)


def main(cfg, result_dir, checkpoint=None, reset_epoch=False):

    os.makedirs(result_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Get git repo version
    meta_info = {'checkpoint': checkpoint}
    repo = git.Repo(search_parent_directories=True)
    meta_info['version'] = repo.head.object.hexsha
    meta_info['git_diff'] = '<pre><code>{}</code></pre>'.format(repo.git.diff('--ignore-submodules'))
    del repo

    # Make sure results are reproducible
    seed()
    train_data = datasets.MNIST('../../data/MNIST', train=True, download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    val_data = datasets.MNIST('../../data/MNIST', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    trainer = TrainOnMNIST(cfg, device, len(train_data.classes), result_dir, train_dataset = train_data, val_dataset = val_data, meta=meta_info)

    if checkpoint:
        trainer.load_checkpoint(checkpoint, reset_epoch=reset_epoch)

    trainer.train()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train net on MNIST data')
    parser.add_argument('--cfg', type=str, help='Path to config file. Default: %(default)s', default=osp.join('..','cfg','train_mnist.yaml'))
    parser.add_argument('--result-dir', type=str, help='Path to result. Default: %(default)s', default = osp.join('..','..','data','runs','MNIST'))
    parser.add_argument('--checkpoint', type=str, help='Path to weights to start from')
    parser.add_argument('--reset-epoch', action='store_true', help='Reset epoch when loading checkpoint')

    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    #Create log dir
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    result_dir = osp.join(args.result_dir, current_time)

    main(cfg, result_dir, args.checkpoint, args.reset_epoch)
