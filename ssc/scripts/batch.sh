#!/usr/bin/env bash
#------------All data included----------------
#Train
# python3 train.py /host_home/data/suncg/scene_comp/data3 \
#                  /host_home/data/suncg/scene_comp/datasets/train_mini.json \
#                  --val /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#                  --cfg ../cfg/train.yaml \
#                  --result-dir ../../data/runs/ICPR2/suncg11_det_wd0_data3
#
# python3 train.py /host_home/data/suncg/scene_comp/data3 \
#                  /host_home/data/suncg/scene_comp/datasets/train_mini.json \
#                  --val /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#                  --cfg ../cfg/train2.yaml \
#                  --result-dir ../../data/runs/ICPR2/suncg11_det_wd0.01_data3

#Eval
# python3 eval.py /host_home/data/suncg/scene_comp/data3 \
#                 /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#                 ../../data/runs/ICPR2_train/suncg11_det_wd0_data3/Jun25_17-15-44 \
#                 --result-dir ../../data/runs/ICPR2_eval/suncg11_det_wd0_data3
#                 --cfg ../cfg/eval.yaml
#
# python3 eval.py /host_home/data/suncg/scene_comp/data3 \
#                 /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#                 ../../data/runs/ICPR2_train/suncg11_det_wd0.05_data3/Jun26_01-27-08 \
#                 --result-dir ../../data/runs/ICPR2_eval/suncg11_det_wd0.05_data3 \
#                 --cfg ../cfg/eval.yaml
#
# python3 eval.py /host_home/data/suncg/scene_comp/data3 \
#                 /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#                 ../../data/runs/ICPR/suncg11_bssc_data3_Apr16_11-53-56 \
#                 --result-dir ../../data/runs/ICPR2_eval/suncg11_bssc_data3 \
#                 --cfg ../cfg/eval_bayesian.yaml

#------------No bed in training----------------
#Train
# python3 train.py /host_home/data/suncg/scene_comp/data4_no_bed \
#                  /host_home/data/suncg/scene_comp/datasets/train_mini.json \
#                  --val /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#                  --cfg ../cfg/train2.yaml \
#                  --result-dir ../../data/runs/ICPR2/suncg11_det_wd0.01_data4_nobed
#Eval
# python3 eval.py /host_home/data/suncg/scene_comp/data3 \
#                 /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#                 ../../data/runs/ICPR/suncg11_bssc_data4_nobed_Apr22_08-14-34 \
#                 --cfg ../cfg/eval_bayesian.yaml \
#                 --result-dir ../../data/runs/ICPR2_eval/suncg11_bssc_data3_data4_nobed_model
#
# python3 eval.py /host_home/data/suncg/scene_comp/data3 \
#                 /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#                 ../../data/runs/ICPR/suncg11_det_wd0_data4_nobed_Apr22_18-59-04 \
#                 --cfg ../cfg/eval.yaml \
#                 --result-dir ../../data/runs/ICPR2_eval/suncg11_det_wd0_data3_data4_nobed_model

# python3 eval.py /host_home/data/suncg/scene_comp/data3 \
#                 /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#                 ../../data/runs/ICPR/suncg11_det_wd0.05_data4_nobed_Apr23_03-49-13 \
#                 --cfg ../cfg/eval.yaml \
#                 --result-dir ../../data/runs/ICPR2_eval/suncg11_det_wd0.05_data3_data4_nobed_model


# python3 eval.py /host_home/data/suncg/scene_comp/data3 \
#                 /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#                 ../../data/runs/ICPR2/suncg11_det_wd0.01_data4_nobed/Jul05_08-18-00 \
#                 --cfg ../cfg/eval.yaml \
#                 --result-dir ../../data/runs/ICPR2_eval/suncg11_det_wd0.01_data3_data4_nobed_model

#----------------------Train regression classifer--------------
#Train
# python3 train.py /host_home/data/suncg/scene_comp/data3 \
#                  /host_home/data/suncg/scene_comp/datasets/train_mini.json \
#                  --val /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#                  --cfg ../cfg/train_bayesian_sampler.yaml \
#                  --result-dir ../../data/runs/ICPR2/suncg11_bssc_sampler_data3 \
#                  --checkpoint ../../data/runs/ICPR/suncg11_bssc_data3_Apr16_11-53-56/ckp_00300.tar \
#                  --reset-epoch
#
# python3 train_mnist.py \
#                  --cfg ../cfg/train_mnist_sampler.yaml \
#                  --result-dir ../../data/runs/MNIST/sample_nonlinear \
#                  --checkpoint ../../data/runs/MNIST/full_bnn_0.5_Jul01_14-26-18/ckp_00060.tar \
#                  --reset-epoch

 # python3 train.py /host_home/data/suncg/scene_comp/data3 \
 #                  /host_home/data/suncg/scene_comp/datasets/train_mini.json \
 #                  --val /host_home/data/suncg/scene_comp/datasets/val_mini.json \
 #                  --cfg ../cfg/train_bayesian_sampler.yaml \
 #                  --result-dir ../../data/runs/ICPR2/suncg11_bssc_unet_sampler_data3 \
 #                  --checkpoint ../../data/runs/ICPR2/suncg11_bssc_unet_data3/Jul07_20-31-36_sweep/'kl_beta:1|prior:{name:cauchy,gamma:0.1}|var_init:0.05|'/ckp_00200.tar \
 #                  --reset-epoch


#----------------------UNet--------------------
# python3 sweep_params.py  /host_home/data/suncg/scene_comp/data3 \
#   /host_home/data/suncg/scene_comp/datasets/train_mini.json \
#   --val /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#   --base-cfg ../cfg/train_bayesian_unet.yaml \
#   --sweep-cfg ../cfg/sweep_unet.yaml \
#   --result-dir ../../data/runs/ICPR2/suncg11_bssc_unet_data3/

# Fixed
# python3 sweep_params.py  /host_home/data/suncg/scene_comp/data3 \
#   /host_home/data/suncg/scene_comp/datasets/train_mini.json \
#   --val /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#   --base-cfg ../cfg/train_det_unet.yaml \
#   --sweep-cfg ../cfg/sweep_wd.yaml \
#   --result-dir ../../data/runs/ICPR2/suncg11_det_unet_data3/

#Nobed bayesian + fixed

# Beta Sweep with wider distribution

# Whole dataset?

# Unet nobed
# python3 sweep_params.py  /host_home/data/suncg/scene_comp/data4_no_bed \
#   /host_home/data/suncg/scene_comp/datasets/train_mini.json \
#   --val /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#   --base-cfg ../cfg/train_det_unet.yaml \
#   --sweep-cfg ../cfg/sweep_wd.yaml \
#   --result-dir '../../data/runs/ICPR2/suncg11|net:det_unet|train:data4_nobed|val:data4_nobed|'

# Eval nobed for Bayesian Unet
# python3 eval.py /host_home/data/suncg/scene_comp/data3 \
#                 /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#                 '../../data/runs/ICPR2/suncg11|net:bssc_unet|kl_beta:5|train:data4_nobed|val:data4_nobed|/Jul13_10-02-05' \
#                 --cfg ../cfg/eval_bayesian_unet.yaml \
#                 --result-dir '../../data/runs/ICPR2_eval/suncg11|net:bssc_unet|kl_beta:5|train:data4_nobed|val:data3|'

# python3 eval.py /host_home/data/suncg/scene_comp/data3 \
#                 /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#                 '../../data/runs/ICPR2/suncg11|net:det_unet|train:data4_nobed|val:data4_nobed|/Jul13_22-14-04_sweep/weight_decay:0|' \
#                 --cfg ../cfg/eval_unet.yaml \
#                 --result-dir '../../data/runs/ICPR2_eval/suncg11|net:det_unet|wd:0|train:data4_nobed|val:data3|'
#
# python3 eval.py /host_home/data/suncg/scene_comp/data3 \
#                 /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#                 '../../data/runs/ICPR2/suncg11|net:det_unet|train:data4_nobed|val:data4_nobed|/Jul13_22-14-04_sweep/weight_decay:0.01|' \
#                 --cfg ../cfg/eval_unet.yaml \
#                 --result-dir '../../data/runs/ICPR2_eval/suncg11|net:det_unet|wd:0.01|train:data4_nobed|val:data3|'


#MNIST all labels
# python3 sweep_params_mnist.py \
#   --base-cfg ../cfg/train_mnist.yaml \
#   --sweep-cfg ../cfg/sweep_wd.yaml \
#   --result-dir ../../data/runs/MNIST/all_labels

# python3 train_mnist.py \
#   --cfg ../cfg/train_mnist_bayesian.yaml \
#   --result-dir ../../data/runs/MNIST/all_labels

# Generate example images
python3 eval.py /host_home/data/suncg/scene_comp/data3 \
                /host_home/data/suncg/scene_comp/datasets/val_mini.json \
                ../../data/runs/ICPR/suncg11_bssc_data4_nobed_Apr22_08-14-34/ckp_00150.tar \
                --cfg ../cfg/eval_bayesian.yaml \
                --result-dir '../../data/runs/ICPR2_eval/examples/suncg11|net:bssc|train:data4_nobed|val:data3|crossection|'
#
# python3 eval.py /host_home/data/suncg/scene_comp/data3 \
#                 /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#                 ../../data/runs/ICPR/suncg11_det_wd0_data4_nobed_Apr22_18-59-04/ckp_00100.tar \
#                 --cfg ../cfg/eval.yaml \
#                 --result-dir '../../data/runs/ICPR2_eval/examples/suncg11|net:det_ssc|wd:0|train:data4_nobed|val:data3'
#
# python3 eval.py /host_home/data/suncg/scene_comp/data3 \
#                 /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#                 ../../data/runs/ICPR/suncg11_det_wd0_data4_nobed_Apr22_18-59-04/ckp_00100.tar \
#                 --cfg ../cfg/eval.yaml \
#                 --result-dir '../../data/runs/ICPR2_eval/examples/suncg11|net:det_ssc|wd:0|train:data4_nobed|val:data3'
#
# python3 eval.py /host_home/data/suncg/scene_comp/data3 \
#                 /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#                 '../../data/runs/ICPR2/suncg11|net:bssc_unet|kl_beta:5|train:data4_nobed|val:data4_nobed|/Jul13_10-02-05/ckp_00400.tar' \
#                 --cfg ../cfg/eval_bayesian_unet.yaml \
#                 --result-dir '../../data/runs/ICPR2_eval/examples/suncg11|net:bssc_unet|kl_beta:5|train:data4_nobed|val:data3|'

# python3 train.py /host_home/data/suncg/scene_comp/data3 \
#                  /host_home/data/suncg/scene_comp/datasets/train_mini.json \
#                  --val /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#                  --cfg ../cfg/train_bayesian_unet.yaml \
#                  --result-dir '../../data/runs/ICPR2/suncg11|net:bssc_unet|kl_beta:5|train:data3|val:data3|' \
#                  --checkpoint '../../data/runs/ICPR2/suncg11_bssc_unet_data3/Jul10_03-48-22_sweep/kl_beta:5|prior:{name:cauchy,gamma:0.1}|var_init:0.1|/ckp_00400.tar'
#
#  python3 train.py /host_home/data/suncg/scene_comp/data4_no_bed \
#                   /host_home/data/suncg/scene_comp/datasets/train_mini.json \
#                   --val /host_home/data/suncg/scene_comp/datasets/val_mini.json \
#                   --cfg ../cfg/train_bayesian_unet.yaml \
#                   --result-dir '../../data/runs/ICPR2/suncg11|net:bssc_unet|kl_beta:5|train:data4_nobed|val:data4_nobed|' \
#                   --checkpoint '../../data/runs/ICPR2/suncg11|net:bssc_unet|kl_beta:5|train:data4_nobed|val:data4_nobed|/Jul13_10-02-05/ckp_00400.tar'

#python3 train.py /host_home/data/suncg/scene_comp/data4_no_bed/ /host_home/data/suncg/scene_comp/datasets/train_mini.json --val /host_home/data/suncg/scene_comp/datasets/val_mini.json --cfg ../cfg/train_bayesian.yaml --checkpoint ../../data/runs/ICPR/suncg11_bssc_data4_nobed_Apr22_08-14-34/ckp_00150.tar
# python3 train.py /host_home/data/suncg/scene_comp/data4_no_bed/ /host_home/data/suncg/scene_comp/datasets/train_mini.json --val /host_home/data/suncg/scene_comp/datasets/val_mini.json --cfg ../cfg/train.yaml
# python3 train.py /host_home/data/suncg/scene_comp/data4_no_bed/ /host_home/data/suncg/scene_comp/datasets/train_mini.json --val /host_home/data/suncg/scene_comp/datasets/val_mini.json --cfg ../cfg/train2.yaml
