#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
python main.py \
  --config DCGAN/configs/dcgan_celeba64.yaml \
  --command Celeba64 \
  --outdir results/temp/celeba64_inception_moments


# wgan_gp using DCGAN net on Celeba64
python main.py \
  --config DCGAN/configs/dcgan_celeba64.yaml \
  --command wgan_gp_celeba64 \
  --outdir results/temp/wgan_gp_dcgan_celeba64

