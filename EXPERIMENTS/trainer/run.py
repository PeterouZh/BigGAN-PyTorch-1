import os
import pprint
from easydict import EasyDict

def ImageNet128_make_hdf5(args, myargs):
  import make_hdf5
  parser = make_hdf5.prepare_parser()
  config = vars(parser.parse_args())
  config = EasyDict(config)

  config1 = myargs.config.ImageNet128_make_hdf5
  for k, v in config1.items():
    setattr(config, k, v)
  config.data_root = os.path.expanduser(config.data_root)
  myargs.logger.info(pprint.pformat(config))
  make_hdf5.run(config)
  pass


def main(args, myargs):
  exec('%s(args, myargs)'%args.command)