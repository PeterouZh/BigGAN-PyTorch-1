# Prepare inception moment file
```bash
export CUDA_VISIBLE_DEVICES=0
python main.py \
  --config ./configs/prepare_data.yaml \
  --command Calculate_inception_moments_Celeba64 \
  --outdir results/temp/Calculate_inception_moments_Celeba64
```
# Train
## wgan-gp on celeba64
```bash
export CUDA_VISIBLE_DEVICES=0
python main.py \
  --config DCGAN/configs/dcgan_celeba64.yaml \
  --command CelebA64_dcgan_wgan_gp \
  --outdir results/temp/CelebA64_dcgan_wgan_gp
```
## wbgan-gp on celeba64
```bash
export CUDA_VISIBLE_DEVICES=0
python main.py \
  --config DCGAN/configs/dcgan_celeba64.yaml \
  --command CelebA64_dcgan_wgan_gp_bound_sinkhorn \
  --outdir results/temp/CelebA64_dcgan_wgan_gp_bound_sinkhorn
```