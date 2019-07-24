import os

from template_lib.datasets import CelebA

def get_data_loaders(config):

  if config.dataset == 'Celeba64':
    data_loader = CelebA.CelebA64(
      datadir=os.path.expanduser(config.datadir),
      batch_size=config.batch_size,
      shuffle=config.shuffle,
      num_workers=config.num_workers,
      seed=config.seed)
  else:
    assert 0
  return data_loader
