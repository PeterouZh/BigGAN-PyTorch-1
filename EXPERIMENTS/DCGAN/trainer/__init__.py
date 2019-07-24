from . import wgan_gp_trainer


exe_dict = {
  "wgan_gp_celeba64": "train",
  "wbgan_gp_celeba64": "train"
}

trainer_dict = {
  "wgan_gp_celeba64": wgan_gp_trainer.Trainer,
  "wbgan_gp_celeba64": wgan_gp_trainer.Trainer
}

run_dict = {

}