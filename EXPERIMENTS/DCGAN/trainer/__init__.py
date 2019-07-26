from . import wgan_gp_trainer, wgan_gp_trainer_distributed


exe_dict = {
  "wgan_gp_celeba64": "train",
  "wbgan_gp_celeba64": "train",
  "wbgan_gp_dist_celeba64": "train_dist"
}

trainer_dict = {
  "wgan_gp_celeba64": wgan_gp_trainer.Trainer,
  "wbgan_gp_celeba64": wgan_gp_trainer.Trainer,
  "wbgan_gp_dist_celeba64": wgan_gp_trainer_distributed.Trainer
}

run_dict = {

}