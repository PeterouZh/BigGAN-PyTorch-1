


exe_dict = {
  "wgan_gp_celeba64": "train",
  "wbgan_gp_celeba64": "train",
  "wbgan_gp_dist_celeba64": "train_dist",
}

from . import wgan_gp_trainer, wgan_gp_trainer_distributed
from . import wgan_gpreal_trainer

trainer_dict = {
  "wgan_gp_celeba64": wgan_gp_trainer.Trainer,
  "wbgan_gp_celeba64": wgan_gp_trainer.Trainer,
  "wbgan_gp_dist_celeba64": wgan_gp_trainer_distributed.Trainer,
  'wgan_gpreal_celeba64': wgan_gpreal_trainer.Trainer,
  'wbgan_gpreal_celeba64': wgan_gpreal_trainer.Trainer
}

run_dict = {

}