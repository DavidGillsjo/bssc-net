#!/usr/bin/python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import argparse
import os.path as osp
from torch.utils.data import DataLoader
from ssc.data.loader import SUNCGDataset
from datetime import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
import git
from importlib import import_module
from ssc.net.bayesian.layers import BNLLLoss, BConv
from ssc.net.bayesian.models import SampleNet

import ssc.visualization.mayavi_voxel as mviz
from ssc.visualization.histogram import HistogramGroup
from ssc.visualization.weight import plot_sample_net
from ssc.visualization.pr import PRGroup
from ssc.utils.metrics import *
import itertools
import hashlib
import cv2


def dict2md(my_dict):
    table = '|Parameter|Value|\n|---------|-----|'
    for param in sorted(my_dict):
        table += '\n|{}|{}|'.format(param,my_dict[param])
    return table


def seed():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TrainNet:
    '''
    Supply train dataset to be in training mode.
    Only val_dataset will make it run in test mode.
    '''
    def __init__(self, cfg, device, nbr_classes, result_dir, train_dataset = None, val_dataset = None, meta = None):
        self.cfg = cfg
        self.result_dir = result_dir
        #Load net
        module_name, _, class_name = cfg['net'].rpartition('.')
        net_class = getattr(import_module(module_name), class_name)
        net = net_class(nbr_classes, cfg)
        self.bayesian = getattr(net, 'bayesian', False)

        if self.bayesian:
            self.net = SampleNet(net, cfg).to(device)
            opt_params = [{'params': self.net.bnn.parameters()}]
            if self.net.metrics:
                opt_params.append({'params': self.net.sample_net.parameters(), 'weight_decay': cfg.get('sample_net_weight_decay', 0)})
        else:
            self.net = net.to(device)
            opt_params = self.net.parameters()

        self.nbr_tb_examples = cfg.get('nbr_tb_examples', 0)
        self.nbr_examples = max(cfg.get('nbr_examples', 0), self.nbr_tb_examples)
        self.examples_dir = osp.join(self.result_dir, 'examples')
        os.mkdir(self.examples_dir)

        if self.bayesian and cfg.get('fixed', False):
            self.net.fix_parameters()

        # Training setup
        self.nclass = net.nbr_classes
        self.device = device
        loader_kwargs = dict(batch_size=self.cfg['batch_size'], shuffle=cfg['shuffle'], num_workers=self.cfg['num_workers'], pin_memory=True)
        self.train_dataloader = DataLoader(train_dataset, **loader_kwargs) if train_dataset else None
        self.val_dataloader = DataLoader(val_dataset, **loader_kwargs) if val_dataset else None
        self.train_mode = train_dataset is not None

        example_cat = self.cfg.get('example_categories', None)
        if example_cat:
            dataloader = self.train_dataloader if self.train_mode else self.val_dataloader
            self.example_cat_ids = [dataloader.dataset.get_class_id(e) for e in example_cat]
        else:
            self.example_cat_ids = None

        meta['nbr_parameters'] = count_parameters(self.net)
        nbr_weights = meta['nbr_parameters']/2 if self.bayesian else meta['nbr_parameters']
        self.epoch = 0
        self.first_epoch = True

        if self.train_mode:
            self.criterion = BNLLLoss(cfg['kl_beta'], len(self.train_dataloader), nbr_weights)
            self.optimizer = Adam(opt_params, lr=self.cfg['learning_rate'], weight_decay=cfg['weight_decay'])

        #Save config
        with open(osp.join(self.result_dir, 'config.yaml'), 'w') as f:
            yaml.dump({'config': self.cfg, 'meta': meta}, f, default_flow_style=False)

        #Setup tensorboard loggers
        self.tblogger = {}
        if train_dataset:
            self.tblogger['train'] = SummaryWriter(osp.join(self.result_dir,'train'))
        if val_dataset:
            self.tblogger['val'] = SummaryWriter(osp.join(self.result_dir,'val'))

        #Write config to tensorboard
        meta['model'] = '<pre><code>{}</code></pre>'.format(str(self.net))
        for _, tbl in self.tblogger.items():
            tbl.add_text('Config', dict2md(self.cfg), global_step=0)
            tbl.add_text('Meta', dict2md(meta), global_step=0)


    def init_histograms(self, tsdf_bins):
        if self.bayesian:
            self.histograms = HistogramGroup({
                                            'avg_max_scores': [0, 1.0],
                                            'entropy_top': [0, 0.4],
                                            'var_e_top': [0, 0.3],
                                            'var_a_top': [0, 0.3]
                                            },
                                             metric_pairs = [('avg_max_scores', 'entropy_top')],
                                             make_pdf =self.cfg.get('make_pdf', False)
                                             )
        else:
             self.histograms = HistogramGroup({'avg_max_scores': [0, 1.0]}, make_pdf =self.cfg.get('make_pdf', False))

        self.histograms.add_type('semantic', self.val_dataloader.dataset.get_class_labels())
        bins_str = ['[{:.2g}, {:.2g}]'.format(left, right) for left,right in zip(tsdf_bins[:-1], tsdf_bins[1:])]
        self.histograms.add_type('tsdf', bins_str)

    def init_prs(self, tsdf_bins):
        if self.bayesian:
            self.prs = PRGroup(['avg_max_scores', 'entropy_top'],
                              [[0.05, 0.95], [0.05, 0.5]],
                              [0.05, 0.05], [True, False],
                              device=self.device,
                              make_pdf =self.cfg.get('make_pdf', False))
        else:
            self.prs = PRGroup(['avg_max_scores'], [[0.05, 0.95]], [0.05], [True], device=self.device, make_pdf =self.cfg.get('make_pdf', False))


        self.prs.add_type('semantic', self.val_dataloader.dataset.get_class_labels())
        bins_str = ['[{:.2g}, {:.2g}]'.format(left, right) for left,right in zip(tsdf_bins[:-1], tsdf_bins[1:])]
        self.prs.add_type('tsdf', self.val_dataloader.dataset.get_class_labels(), bins_str)


    def train_loop(self):
        assert self.train_mode, 'Training dataset must be supplied for training'

        while True:
            #Train
            self.net.train()
            self.update_learning_rate()
            self.run_epoch()
            self.first_epoch = False

            if self.epoch > 0 and not (self.epoch % self.cfg['save_interval']):
                self.save_checkpoint()

            #Validation set
            if self.val_dataloader and self.epoch > 0 and not (self.epoch % self.cfg['val_interval']):
                self.net.eval()
                with torch.no_grad():
                    self.run_epoch()

            self.epoch += 1
            if self.epoch >= self.cfg['epochs']:
                return


    def train(self):
        try:
            self.train_loop()
        except KeyboardInterrupt:
            #To skip error output
            pass
        finally:
            #Always take a checkpoint when failing (unless we just started)
            if self.epoch > 0:
                self.save_checkpoint()

    def eval_weights(self, weights_path):
        '''
        Should be used when in test mode (no training data supplied) to evaluate a set of model weights on a dataset.
        '''
        self.first_epoch = False
        self.load_checkpoint(weights_path)
        self.net.eval()
        with torch.no_grad():
            self.run_epoch()


    def update_learning_rate(self):
        self.lr = self.cfg['learning_rate'] * (self.cfg['lr_decay_factor'] ** (self.epoch // self.cfg['lr_decay_period']))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr


    def forward(self, data, gt, batch_idx, mask = None):
        result = self.net.forward(data)
        if not self.bayesian:
            result['kl'] = 0

        if self.train_mode:
            # Compute Loss
            if mask is None:
                self.criterion.reduction = 'mean'
                result['loss'] = self.criterion(result['log_score'], result['kl'], gt, batch_idx)
            else:
                self.criterion.reduction = 'none'
                loss_seq = self.criterion(result['log_score'], result['kl'], gt, batch_idx)
                result['loss'] = torch.mean(loss_seq[mask])

        return result

    def run_epoch(self):
        train = self.net.training
        dataloader = self.train_dataloader if train else self.val_dataloader
        tblogger = self.tblogger['train'] if train else self.tblogger['val']
        examples_count = 0

        acc_loss = 0
        acc_kl = 0
        if not train:
            confusion = torch.zeros([dataloader.dataset.nbr_tsdf_hist_bins, self.nclass, self.nclass], device = self.device, dtype = torch.long)
            acc_score = np.zeros(dataloader.dataset.nbr_tsdf_hist_bins)

        nbr_batches = len(dataloader)
        for i_batch, sample_batched in enumerate(dataloader):
            #Load data
            tsdf_batch_cpu = sample_batched[self.cfg['tsdf_type']]
            tsdf_batch = tsdf_batch_cpu.to(self.device)
            gt_batch_cpu = sample_batched['gt']
            gt_batch = gt_batch_cpu.to(self.device)

            #Load masks
            visible_free = sample_batched['visible_free'].to(self.device)
            frustum_mask = sample_batched['frustum_mask'].to(self.device)
            occluded_mask = sample_batched['occluded_mask'].to(self.device)
            loss_mask = sample_batched['loss_mask'].to(self.device)

            #Params
            batch_size = gt_batch.shape[0]
            first_val = (i_batch == 0) and not train

            #Aassume TSDF volumes are generated the same way throughout the batch
            if first_val:
                self.init_histograms(sample_batched['tsdf_hist_bins'][0])
                self.init_prs(sample_batched['tsdf_hist_bins'][0])

            #Forward pass and loss calculation
            result = self.forward(tsdf_batch, gt_batch, i_batch, mask = loss_mask)

            log_score = result['log_score']

            # Batch Metrics
            if self.train_mode:
                loss = result['loss']
                kl = result['kl']
                batch_nbr = self.epoch*len(dataloader) + i_batch
                acc_loss += float(loss.item())
                acc_kl += float(kl)
                tblogger.add_scalar('batch_loss', loss.item(), global_step = batch_nbr)
                tblogger.add_scalar('batch_kl', kl, global_step = batch_nbr)


            if train:
                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                #Validation metrics
                score, pred_batch = log_score.max(1)
                tsdf_hist_masks = sample_batched['tsdf_hist_masks'].to(self.device)

                #Set known freespace
                pred_batch[visible_free] = 0
                score[visible_free] = 0

                # Bin result according to TSDF input
                for i in range(tsdf_hist_masks.shape[1]):
                    hmask = tsdf_hist_masks[:,i]
                    masked_confusion = compute_confusion(pred_batch[hmask], gt_batch[hmask], self.nclass)
                    confusion[i] += masked_confusion
                    acc_score[i] += score[hmask].exp().sum()

                if 'mean_score' in result:
                    avg_max_scores, max_idx = result['mean_score'].max(1)
                    result['avg_max_scores'] = avg_max_scores

                    #Set known freespace
                    max_idx[visible_free] = 0
                    avg_max_scores[visible_free] = 1.0

                    #Create occupied stats
                    free_score = result['mean_score'][:,0]
                    free_score[visible_free] = 1.0
                    free_entropy = result['entropy'][:,0]
                    free_variance = result['var'][:,0]

                    max_idx_gindex = max_idx[:,None]
                    result['var_a_top'] = torch.gather(result['var_aleatoric'], 1, max_idx_gindex).squeeze()
                    result['var_e_top'] = torch.gather(result['var_epistemic'], 1, max_idx_gindex).squeeze()
                    result['var_top'] = result['var_a_top'] + result['var_e_top']
                    result['entropy_top'] = torch.gather(result['entropy'], 1, max_idx_gindex).squeeze()

                    corr_mask = (pred_batch == gt_batch)
                    for key in ['var_a_top', 'var_e_top', 'var_top', 'entropy_top']:
                        tblogger.add_histogram('{} - Correct'.format(key), result[key][corr_mask], self.epoch)
                        tblogger.add_histogram('{} - Incorrect'.format(key), result[key][~corr_mask], self.epoch)

                else:
                    free_score = log_score[:,0].exp()
                    result['avg_max_scores'] = avg_max_scores = score.exp()

                self.histograms.add(pred_batch, gt_batch, result, {'tsdf': tsdf_hist_masks})
                self.prs.add(pred_batch, gt_batch, result, {'tsdf': tsdf_hist_masks})

            # Plot graph on first epoch and batch
            # if self.first_epoch and i_batch == 0:
            #     if self.bayesian and not self.net.fixed:
            #         self.net.fix_parameters()
            #         tblogger.add_graph(self.net, tsdf_batch)
            #         self.net.release_parameters()
            #     else:
            #         tblogger.add_graph(self.net, tsdf_batch)

            # Plot examples
            if not train and examples_count < self.nbr_examples:
                #Remove data from GPU to make room for rendering
                pred_batch_cpu = pred_batch.to('cpu')
                del gt_batch, tsdf_batch, pred_batch
                exp_img = {}

                for ex_idx in range(min(self.nbr_examples - examples_count, batch_size)):

                    if self.example_cat_ids and not np.any(np.isin(gt_batch_cpu[ex_idx].numpy(), self.example_cat_ids)):
                        continue

                    ex_mask = occluded_mask[ex_idx].to('cpu').numpy()
                    ex_vox_min = sample_batched['vox_min'][ex_idx].numpy()
                    ex_vox_unit = sample_batched['vox_unit'][ex_idx].numpy()
                    ex_cam_P = sample_batched['cam_P'][ex_idx].numpy()
                    exp_img['Semantic'] = mviz.compare_voxels({'Predicted': pred_batch_cpu[ex_idx].numpy(),
                                                   'GT': gt_batch_cpu[ex_idx].numpy()},
                                                  ex_vox_min, ex_vox_unit, mask = ex_mask,
                                                  camera_P = ex_cam_P, suncg_labels = dataloader.dataset.get_class_labels(),
                                                  crossection=self.cfg.get('crossection', False))

                    exp_img['Score'] = mviz.compare_voxels({'Score': avg_max_scores[ex_idx].to('cpu').numpy(),
                                             'Free score': free_score[ex_idx].to('cpu').numpy()},
                                            ex_vox_min, ex_vox_unit, mask = ex_mask,
                                            cmap = 'gray', camera_P = ex_cam_P,
                                            vmin = 0, vmax = 1, scalar = True,
                                            crossection=self.cfg.get('crossection', False))

                    tsdf_np = sample_batched['tsdf'][ex_idx].squeeze().numpy()
                    flipped_tsdf_np = sample_batched['flipped_tsdf'][ex_idx].squeeze().numpy()
                    # tsdf_mask = (np.abs(flipped_tsdf_np) > 0) & sample_batched['frustum_mask'][ex_idx].numpy()
                    tsdf_mask = (np.abs(tsdf_np) < 1) & sample_batched['frustum_mask'][ex_idx].numpy()
                    # tsdf_mask = None
                    exp_img['TSDF'] = mviz.compare_voxels({'TSDF': tsdf_np,'Flipped TSDF': flipped_tsdf_np},
                                            ex_vox_min, ex_vox_unit, mask = tsdf_mask,
                                            cmap = 'jet', camera_P = ex_cam_P, scalar = True, vmin = -1, vmax = 1, alpha=1.0,
                                            crossection=self.cfg.get('crossection', False))

                    if 'mean_score' in result:
                        exp_img['Entropy'] = mviz.compare_voxels({'Entropy': result['entropy_top'][ex_idx].to('cpu').numpy(),
                                                 'Free Entropy': free_entropy[ex_idx].to('cpu').numpy()},
                                                ex_vox_min, ex_vox_unit, mask = ex_mask,
                                                cmap = 'jet', camera_P = ex_cam_P, scalar = True,
                                                crossection=self.cfg.get('crossection', False))

                        exp_img['Variance'] = mviz.compare_voxels({'Variance': result['var_top'][ex_idx].to('cpu').numpy(),
                                                 'Free Variance': free_variance[ex_idx].to('cpu').numpy()},
                                                ex_vox_min, ex_vox_unit, mask = ex_mask,
                                                cmap = 'jet', camera_P = ex_cam_P, scalar = True,
                                                crossection=self.cfg.get('crossection', False))

                    for desc, img in exp_img.items():
                        cv2.imwrite(osp.join(self.examples_dir, '{}_{}.png'.format(desc, examples_count)), img[:,:,::-1])
                        if examples_count < self.nbr_tb_examples:
                            tblogger.add_image("Example {} - {}".format(examples_count, desc), img, global_step = self.epoch, dataformats='HWC')
                    examples_count += 1

        if self.train_mode:
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
                            tblogger.add_figure("Sample weights - {}".format(m_name), sample_w_fig, global_step = self.epoch, close = True)
            tblogger.add_histogram('all_weights', torch.cat(all_weights), self.epoch)
            if all_weights_mu:
                tblogger.add_histogram('all_weights_mu', torch.cat(all_weights_mu), self.epoch)
                tblogger.add_histogram('all_weights_sigma', torch.cat(all_weights_sigma), self.epoch)

        else:
            all_confusion = torch.sum(confusion, dim=0).to('cpu').numpy()

            all_score = acc_score.sum()
            total_nbr_voxels = all_confusion.sum()
            accuracy = 100.0 * float(np.trace(all_confusion))/float(total_nbr_voxels)
            tblogger.add_scalar('accuracy', accuracy, global_step = self.epoch)
            tblogger.add_scalar('avg_score', all_score/float(total_nbr_voxels), global_step = self.epoch)

            conf_fig = plot_confusion(all_confusion, dataloader.dataset.get_class_labels())
            tblogger.add_figure("Confusion Matrix", conf_fig, global_step = self.epoch, close = True)

            IoU = compute_IoU(all_confusion)
            tblogger.add_scalar('mIoU', np.mean(IoU), global_step = self.epoch)
            iou_dict = {cls_name: np.mean(compute_IoU(conf)) for (cls_name, conf) in zip(self.histograms.get_classes('tsdf'), confusion.to('cpu').numpy())}
            tblogger.add_scalars('IoU_TDF', iou_dict, global_step = self.epoch)

            self.histograms.plot_tb(tblogger, self.epoch)
            self.histograms.plot_bhattacharyya_tb(tblogger, self.epoch)

            self.prs.plot_tb(tblogger, self.epoch)


    def save_checkpoint(self):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),

        }, osp.join(self.result_dir, 'ckp_{:05d}.tar'.format(self.epoch)))


    def load_checkpoint(self, ckpt_path, transfer = False, reset_epoch = False):
        ckpt = torch.load(ckpt_path)
        self.net.load_state_dict(ckpt['model_state_dict'], transfer = transfer)

        if not (transfer or reset_epoch):
            self.epoch = ckpt['epoch']
        if self.train_mode:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except ValueError:
                pass

def main(root_dir, json_file, cfg, result_dir, cache_dir,
         val=None, checkpoint=None, transfer=False, reset_epoch=False):

    os.makedirs(result_dir)

    # Get git repo version
    meta_info = locals()
    repo = git.Repo(search_parent_directories=True)
    meta_info['version'] = repo.head.object.hexsha
    meta_info['git_diff'] = '<pre><code>{}</code></pre>'.format(repo.git.diff('--ignore-submodules'))
    del repo

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Make sure results are reproducible
    seed()
    dset_args = (root_dir, cache_dir)
    dset_kwargs = {k:cfg.get(k, None) for k in ['mapping']}
    dset_kwargs['mp_loader'] = cfg['num_workers'] > 0
    suncg_data = SUNCGDataset(json_file, *dset_args, **dset_kwargs)
    val_data = SUNCGDataset(val, *dset_args, **dset_kwargs, val = True) if val else None
    trainer = TrainNet(cfg, device, suncg_data.get_nbr_classes(), result_dir, train_dataset = suncg_data, val_dataset = val_data, meta=meta_info)

    if checkpoint:
        trainer.load_checkpoint(checkpoint, transfer, reset_epoch)

    trainer.train()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train net on SUNCG data')
    parser.add_argument('root_dir', type=str, help='Path to suncg data folder')
    parser.add_argument('json_file', type=str, help='Path to json file describing the dataset')
    parser.add_argument('--cfg', type=str, help='Path to config file. Default: %(default)s', default=osp.join('..','cfg','train.yaml'))
    parser.add_argument('--val', type=str, help='Path to validation set', default=None)
    parser.add_argument('--result-dir', type=str, help='Path to result. Default: %(default)s', default = osp.join('..','..','data','runs'))
    parser.add_argument('--cache-dir', type=str, help='Path to cache. Default: %(default)s', default = osp.join('..','..','data','_cache_'))
    parser.add_argument('--checkpoint', type=str, help='Path to weights to start from')
    parser.add_argument('--transfer', action='store_true', help='Transfer learn from weights, create new softmax layer and skip optimizer state')
    parser.add_argument('--reset-epoch', action='store_true', help='Reset epoch when loading checkpoint')

    args = parser.parse_args()

    #Create log dir
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    result_dir = osp.join(args.result_dir, current_time)

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    main(args.root_dir, args.json_file, cfg, result_dir, args.cache_dir,
         val=args.val, checkpoint=args.checkpoint, transfer=args.transfer, reset_epoch=args.reset_epoch)
