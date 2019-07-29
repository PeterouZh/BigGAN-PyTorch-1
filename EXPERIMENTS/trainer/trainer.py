import functools, time
import collections
import torch
import torchvision
import torch.nn as nn

from template_lib.models import ema_model
from template_lib.gans import gan_utils, inception_utils, gan_losses, \
  weight_regularity
from template_lib.trainer import base_trainer

from TOOLS import dataset
from TOOLS import interpolation_BigGAN


class BaseTrainer(base_trainer.Trainer):
  def __init__(self, args, myargs):
    self.args = args
    self.myargs = myargs
    self.config = myargs.config
    self.logger = myargs.logger
    self.train_dict = self.init_train_dict()

    # self.dataset_load()
    self.model_create()
    self.optimizer_create()
    self.schedule_create()

    self.noise_create()
    # load inception network
    self.inception_metrics = self.inception_metrics_func_create()

  def init_train_dict(self, ):
    train_dict = collections.OrderedDict()
    train_dict['epoch_done'] = 0
    train_dict['batches_done'] = 0
    train_dict['best_FID'] = 9999
    self.myargs.checkpoint_dict['train_dict'] = train_dict
    return train_dict

  def model_create(self):
    import BigGAN as model
    import utils
    config = self.config.model

    Generator = model.Generator
    Discriminator = model.Discriminator
    G_D = model.G_D

    print('Create generator: {}'.format(Generator))
    self.resolution = utils.imsize_dict[self.config.loader.dataset]
    self.n_classes = utils.nclass_dict[self.config.loader.dataset]
    G_activation = utils.activation_dict[config.Generator.G_activation]
    self.G = Generator(**{**config.Generator,
                          'resolution': self.resolution,
                          'n_classes': self.n_classes,
                          'G_activation': G_activation},
                       **self.config.optimizer).cuda()
    print('Create discriminator: {}'.format(Discriminator))
    D_activation = utils.activation_dict[config.Discriminator.D_activation]
    self.D = Discriminator(logger=self.logger,
                           **{**config.Discriminator,
                              'resolution': self.resolution,
                              'n_classes': self.n_classes,
                              'D_activation': D_activation},
                           **self.config.optimizer).cuda()
    self.G_ema = Generator(logger=self.logger,
                           **{**config.Generator,
                              'resolution': self.resolution,
                              'n_classes': self.n_classes,
                              'G_activation': G_activation,
                              'skip_init': True,
                              'no_optim': True}).cuda()
    self.ema = ema_model.EMA(
      self.G, self.G_ema, decay=0.9999, start_itr=config.ema_start)

    print('Create G_D: {}'.format(G_D))
    self.GD = G_D(self.G, self.D)
    if config['parallel']:
      self.GD = nn.DataParallel(self.GD)

    self.myargs.checkpoint_dict['G'] = self.G
    self.myargs.checkpoint_dict['G_optim'] = self.G.optim
    self.myargs.checkpoint_dict['D'] = self.D
    self.myargs.checkpoint_dict['D_optim'] = self.D.optim
    self.myargs.checkpoint_dict['G_ema'] = self.G_ema

    models = {'G': self.G, 'D': self.D}
    self.print_number_params(models=models)

  def optimizer_create(self):
    pass

  def noise_create(self):
    config = self.myargs.config.noise
    G_batch_size = max(config['G_batch_size'], config['batch_size'])
    self.z_ = gan_utils.z_normal(
      batch_size=G_batch_size,
      dim_z=self.config.model.Generator.dim_z,
      z_mean=config.z_mean, z_var=config.z_var)
    self.y_ = gan_utils.y_categorical(batch_size=G_batch_size,
                                      nclasses=self.n_classes)

    self.z_test = gan_utils.z_normal(
      batch_size=G_batch_size,
      dim_z=self.config.model.Generator.dim_z,
      z_mean=config.z_mean, z_var=config.z_var)
    self.y_test = gan_utils.y_categorical(batch_size=G_batch_size,
                                          nclasses=self.n_classes)

    # Prepare a fixed z & y to see individual sample evolution throghout training
    self.fixed_z = gan_utils.z_normal(
      batch_size=G_batch_size,
      dim_z=self.config.model.Generator.dim_z,
      z_mean=config.z_mean, z_var=config.z_var)
    self.fixed_y = gan_utils.y_categorical(batch_size=G_batch_size,
                                           nclasses=self.n_classes)
    self.fixed_z.sample_()
    self.fixed_y.sample_()

  def inception_metrics_func_create(self):
    config = self.config.inception_metric
    self.logger.info('Load inception moments: %s',
                     config.saved_inception_moments)
    inception_metrics = inception_utils.InceptionMetricsCond(
      saved_inception_moments=config.saved_inception_moments)

    inception_metrics = functools.partial(
      inception_metrics,
      num_inception_images=config.num_inception_images,
      num_splits=10, prints=True)

    return inception_metrics

  def dataset_load(self, ):
    if self.config.train_one_epoch.dummy_train:
      return
    config = self.config.dataset
    self.logger.info('Load dataset in: %s', config.data_root)

    D_batch_size = (self.config.noise.batch_size
                    * self.config.train_one_epoch.num_D_steps
                    * self.config.train_one_epoch.num_D_accumulations)

    self.loaders = datasets.get_data_loaders(
      logger=self.logger,
      **{**config,
         'batch_size': D_batch_size,
         'start_itr': self.train_dict['batches_done']})

  def _summary_create(self):
    self.summary = {}
    self.summary_D = {}
    self.summary_wd = {}

  def _summary_scalars(self):
    myargs = self.myargs
    for key in self.summary:
      myargs.writer.add_scalar('train_one_epoch/%s' % key, self.summary[key],
                               self.train_dict['batches_done'])
    myargs.writer.add_scalars('logit_mean', self.summary_D,
                              self.train_dict['batches_done'])
    myargs.writer.add_scalars('wd', self.summary_wd,
                              self.train_dict['batches_done'])

  def _summary_images(self, imgs):
    myargs = self.myargs
    train_dict = self.train_dict
    config = self.config.summary_images

    bs_log = config.bs_log
    n_row = config.n_row

    self.G.eval()
    self.G_ema.eval()
    # x
    merged_img = torchvision.utils.make_grid(imgs[:bs_log], normalize=True,
                                             pad_value=1, nrow=n_row)
    myargs.writer.add_images('imgs', merged_img.view(1, *merged_img.shape),
                             train_dict['batches_done'])
    with torch.no_grad():
      # G
      G_z = self.G(self.fixed_z[:bs_log], self.G.shared(self.fixed_y[:bs_log]))
      merged_img = torchvision.utils.make_grid(G_z, normalize=True,
                                               pad_value=1, nrow=n_row)
      myargs.writer.add_images('G_z', merged_img.view(1, *merged_img.shape),
                               train_dict['batches_done'])
      # G_ema
      G_z = self.G_ema(self.fixed_z[:bs_log],
                       self.G_ema.shared(self.fixed_y[:bs_log]))
      merged_img = torchvision.utils.make_grid(G_z, normalize=True,
                                               pad_value=1, nrow=n_row)
      myargs.writer.add_images('G_ema_z',
                               merged_img.view(1, *merged_img.shape),
                               train_dict['batches_done'])

    self.G.train()

  def sample_interpolation(self, G, name_prefix):
    config=self.config.test.sample_interpolation
    myargs = self.myargs
    train_dict = self.train_dict
    G.eval()

    image_filenames, images = interpolation_BigGAN.sample_sheet(
      G=G, classes_per_sheet=config.classes_per_sheet,
      num_classes=self.n_classes, samples_per_class=10,
      parallel=True, z_=self.fixed_z)
    for key, value in zip(image_filenames, images):
      key = name_prefix + '/' + key
      myargs.writer.add_images(key, value, train_dict['epoch_done'])

    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
      image_filename, image = interpolation_BigGAN.interp_sheet(
        G, num_per_sheet=config.num_per_sheet,
        num_midpoints=config.num_midpoints, num_classes=self.n_classes,
        parallel=True, sheet_number=0,
        fix_z=fix_z, fix_y=fix_y, device='cuda')
      myargs.writer.add_images(name_prefix + '/' + image_filename, image,
                               train_dict['epoch_done'])
    G.train()

  def finetune(self):
    config = self.config.finetune
    if config.type == 'imagenet':
      self.logger.info('Finetune imagenet model.')
      self.G.load_state_dict(state_dict=torch.load(config.G_path))
      self.G_ema.load_state_dict(state_dict=torch.load(config.G_ema_path))
      self.D.load_state_dict(state_dict=torch.load(config.D_path))

  def train(self, ):
    config = self.config.train
    self.modelarts()
    for epoch in range(self.train_dict['epoch_done'], config.epochs):
      self.logger.info('epoch: [%d/%d]' % (epoch, config.epochs))

      self.train_one_epoch()

      self.train_dict['epoch_done'] += 1
      # test
      self.test()
    self.finalize()

  def test(self):
    config = self.config.test
    train_dict = self.myargs.checkpoint_dict['train_dict']

    # Interpolation
    if hasattr(config, 'sample_interpolation'):
      self.sample_interpolation(G=self.G, name_prefix='G')
      self.sample_interpolation(G=self.G_ema, name_prefix='G_ema')

    if config.use_ema:
      G = self.G_ema
    else:
      G = self.G
    IS_mean, IS_std, FID = self.inception_metrics(
      G=G, z=self.z_test, y=self.y_test,
      show_process=False, use_torch=False, parallel=True)
    self.logger.info_msg('IS_mean: %f +- %f, FID: %f', IS_mean, IS_std, FID)
    summary = {'IS_mean': IS_mean, 'IS_std': IS_std, 'FID': FID}
    for key in summary:
      self.myargs.writer.add_scalar('test/' + key, summary[key],
                                    train_dict['epoch_done'])
    if train_dict['best_FID'] > FID:
      train_dict['best_FID'] = FID
      self.myargs.checkpoint.save_checkpoint(
        checkpoint_dict=self.myargs.checkpoint_dict,
        filename='ckpt_epoch_%d_FID_%f.tar' % (
          train_dict['epoch_done'], FID))

    # For huawei modelarts tensorboard
    self.modelarts()

  def modelarts(self, ):
    if hasattr(self.args, 'tb_obs'):
      config = self.config.modelarts
      self.logger.info('Copying tb to tb_obs ...')
      self.myargs.copy_obs(self.args.tbdir, self.args.tb_obs, copytree=True)
      if self.train_dict['epoch_done'] % config.copy_results_every == 0:
        self.logger.info('Copying results to obs ...')
        self.myargs.copy_obs('results', self.args.results_obs, copytree=True)
    return

  def finalize(self):
    self.logger.info_msg('best_FID: %f', self.train_dict['best_FID'])
    self.myargs.checkpoint.save_checkpoint(
      checkpoint_dict=self.myargs.checkpoint_dict,
      filename='ckpt_end.tar')
    self.modelarts()

  def evaluate(self):
    self.logger.info("Evaluating ...")
    config = self.config.evaluate
    self.G.load_state_dict(torch.load(config.G_path))
    self.G_ema.load_state_dict(torch.load(config.G_ema_path))

    IS_mean, IS_std, FID = self.inception_metrics(
      G=self.G, z=self.z_test, y=self.y_test,
      show_process=True, use_torch=False, parallel=config.parallel)
    self.logger.info("Test G:")
    self.logger.info_msg('IS_mean: %f +- %f, FID: %f', IS_mean, IS_std, FID)

    IS_mean, IS_std, FID = self.inception_metrics(
      G=self.G_ema, z=self.z_test, y=self.y_test,
      show_process=True, use_torch=False, parallel=config.parallel)
    self.logger.info("Test G_ema:")
    self.logger.info_msg('IS_mean: %f +- %f, FID: %f', IS_mean, IS_std, FID)
    pass

