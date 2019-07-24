import torch
import tqdm

from template_lib.gans import gan_losses

from . import trainer


class Trainer(trainer.Trainer):
  def __init__(self, args, myargs):
    super(Trainer, self).__init__(args, myargs)

  def train_one_epoch(self, ):
    config = self.config.train_one_epoch
    if config.dummy_train:
      return
    myargs = self.myargs
    train_dict = self.train_dict

    self.G.train()
    self.D.train()

    for i, (imgs, _) in enumerate(tqdm.tqdm(self.data_loader, file=myargs.stdout)):
      train_dict['batches_done'] += 1
      self._summary_create()

      imgs = imgs.cuda()
      bs = imgs.size(0)
      self.z_train.sample_()
      f_imgs = self.G(self.z_train[:bs])

      # train D
      D_r_logit = self.D(imgs)
      D_r_logit_mean = D_r_logit.mean()
      D_f_logit = self.D(f_imgs.detach())
      D_f_logit_mean = D_f_logit.mean()
      self.summary_logit_mean['D_r_logit_mean'] = D_r_logit_mean.item()
      self.summary_logit_mean['D_f_logit_mean'] = D_f_logit_mean.item()

      # Wasserstein-1 Distance
      wd = D_r_logit_mean - D_f_logit_mean
      gp = gan_losses.wgan_gp_gradient_penalty(imgs, f_imgs, self.D)
      if self.args.command in ['wbgan_gp_celeba64']:
        D_loss = -wd + gp * config.gp_lambda + \
                 torch.relu(wd - float(config.bound))
        self.summary_wd['bound'] = config.bound
      else:
        D_loss = -wd + gp * config.gp_lambda
      self.summary_wd['wd'] = wd.item()
      self.summary['gp'] = gp.item()
      self.summary['D_loss'] = D_loss.item()

      self.d_optimizer.zero_grad()
      D_loss.backward()
      self.d_optimizer.step()

      if i % config.n_critic == 0:
        # train G
        self.z_train.sample_()
        f_imgs = self.G(self.z_train)
        D_f_logit = self.D(f_imgs)
        D_f_logit_mean = D_f_logit.mean()
        g_loss_only = - D_f_logit_mean
        G_loss = g_loss_only
        self.summary_logit_mean['G_f_logit_mean'] = D_f_logit_mean.item()
        self.summary['g_loss_only'] = g_loss_only.item()
        self.summary['G_loss'] = G_loss.item()

        self.g_optimizer.zero_grad()
        G_loss.backward()
        self.g_optimizer.step()

        # end iter
        self.ema.update(train_dict['batches_done'])

      if i % config.sample_every == 0:
        # images
        self._summary_images(imgs=imgs)
        # checkpoint
        myargs.checkpoint.save_checkpoint(
          checkpoint_dict=myargs.checkpoint_dict, filename='ckpt.tar')
        # summary
        self._summary_scalars()

      elif train_dict['batches_done'] <= config.sample_start_iter:
        self._summary_scalars()