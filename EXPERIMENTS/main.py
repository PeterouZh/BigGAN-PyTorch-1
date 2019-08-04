import os, pprint, sys
import argparse
from tensorboardX import SummaryWriter
from easydict import EasyDict

sys.path.insert(0, '../submodule')
from template_lib import utils
from template_lib.utils import config, modelarts_utils

from TOOLS import calculate_inception_moments
from DCGAN.trainer import run

exe_dict = {
  'Celeba64': calculate_inception_moments.create_inception_moments,
  'wgan_gp_celeba64': run.train
}


def main():
  myargs = argparse.Namespace()
  parser = utils.args_parser.build_parser()
  args = parser.parse_args()

  config.setup_dirs_and_files(args=args)
  config.setup_logger_and_redirect_stdout(args.logfile, myargs)
  print(pprint.pformat(vars(args)))

  config.setup_config(
    config_file=args.config, saved_config_file=args.configfile,
    myargs=myargs)
  myargs.writer = SummaryWriter(logdir=args.tbdir)

  modelarts_utils.modelarts_setup(args, myargs)

  config.setup_checkpoint(ckptdir=args.ckptdir, myargs=myargs)

  args = EasyDict(vars(args))
  myargs.config = EasyDict(myargs.config)

  exe_dict[args.command](args=args, myargs=myargs)

  return


if __name__ == '__main__':
  main()