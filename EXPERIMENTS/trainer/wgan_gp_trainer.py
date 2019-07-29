import tqdm
import torch

from template_lib.gans import gan_utils, gan_losses, weight_regularity

from . import trainer
from TOOLS import utils_BigGAN


class Trainer(trainer.BaseTrainer):

  def train_one_epoch(self, ):
    config = self.config.trainer.train_one_epoch
    if config.dummy_train:
      return
    myargs = self.myargs
    train_dict = myargs.checkpoint_dict['train_dict']
    batch_size = self.config.trainer.noise.batch_size

    for i, (images, labels) in enumerate(tqdm.tqdm(self.loaders[0])):
      # Increment the iteration counter
      train_dict['batches_done'] += 1
      self._summary_create()
      # Make sure G and D are in training mode, just in case they got set to eval
      # For D, which typically doesn't have BN, this shouldn't matter much.
      self.G.train()
      self.D.train()
      self.G_ema.train()
      images, labels = images.cuda(), labels.cuda()

      # Optionally toggle D and G's "require_grad"
      if config.toggle_grads:
        gan_utils.toggle_grad(self.D, True)
        gan_utils.toggle_grad(self.G, False)

      # How many chunks to split x and y into?
      x = torch.split(images, batch_size)
      y = torch.split(labels, batch_size)
      counter = 0
      for step_index in range(config.num_D_steps):
        self.G.optim.zero_grad()
        self.D.optim.zero_grad()
        # If accumulating gradients, loop multiple times before an optimizer step
        for accumulation_index in range(config.num_D_accumulations):
          self.z_.sample_()
          self.y_.sample_()
          D_fake, D_real, G_z = self.GD(
            self.z_[:batch_size],
            self.y_[:batch_size],
            x[counter], y[counter], train_G=False,
            split_D=config.split_D)

          # Compute components of D's loss, average them, and divide by
          # the number of gradient accumulations
          D_real_mean, D_fake_mean, wd, _ = gan_losses.wgan_discriminator_loss(
            r_logit=D_real, f_logit=D_fake)
          gp = gan_losses.wgan_gp_gradient_penalty_cond(
            x=x[counter], G_z=G_z, gy=self.y_[:batch_size], f=self.D)

          D_loss = -wd + gp * config.gp_lambda
          D_losses = D_loss / float(config.num_D_accumulations)
          # Accumulate gradients
          D_losses.backward()
          counter += 1
          self.summary_D['D_real_mean'] = D_real_mean.item()
          self.summary_D['D_fake_mean'] = D_fake_mean.item()
          self.summary_wd['wd'] = wd.item()
          self.summary['gp'] = gp.item()
          self.summary['D_loss'] = D_loss.item()

        # End accumulation
        # Optionally apply ortho reg in D
        if config.D_ortho > 0.0:
          # Debug print to indicate we're using ortho reg in D.
          print('using modified ortho reg in D')
          weight_regularity.ortho(self.D, config.D_ortho)

        self.D.optim.step()

      # Optionally toggle "requires_grad"
      if config.toggle_grads:
        gan_utils.toggle_grad(self.D, False)
        gan_utils.toggle_grad(self.G, True)

      # Zero G's gradients by default before training G, for safety
      self.G.optim.zero_grad()
      # If accumulating gradients, loop multiple times
      for accumulation_index in range(config.num_G_accumulations):
        self.z_.sample_()
        self.y_.sample_()
        D_fake, G_z = self.GD(self.z_, self.y_, train_G=True, return_G_z=True,
                              split_D=config.split_D)

        G_fake_mean, _ = gan_losses.wgan_generator_loss(f_logit=D_fake)
        G_loss = - G_fake_mean
        G_losses = G_loss / float(config['num_G_accumulations'])
        # Accumulate gradients
        G_losses.backward()

        self.summary_D['G_fake_mean'] = G_fake_mean.item()
        self.summary['G_loss'] = G_loss.item()

      # Optionally apply modified ortho reg in G
      if config.G_ortho > 0.0:
        # Debug print to indicate we're using ortho reg in G
        print('using modified ortho reg in G')
        # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
        weight_regularity.ortho(
          self.G, config.G_ortho,
          blacklist=[param for param in self.G.shared.parameters()])
      self.G.optim.step()

      # If we have an ema, update it, regardless of if we test with it or not
      self.ema.update(train_dict['batches_done'])

      if i % config.sample_every == 0:
        self._summary_scalars()

        # singular values
        G_sv_dict = utils_BigGAN.get_SVs(self.G, 'G')
        D_sv_dict = utils_BigGAN.get_SVs(self.D, 'D')
        myargs.writer.add_scalars('sv/G_sv_dict', G_sv_dict,
                                  train_dict['batches_done'])
        myargs.writer.add_scalars('sv/D_sv_dict', D_sv_dict,
                                  train_dict['batches_done'])

        # samples
        self._summary_images(imgs=x[0])

        # weights
        self.save_checkpoint(filename='ckpt.tar')
      elif train_dict['batches_done'] <= config.sample_start_iter:
        # scalars
        self._summary_scalars()





