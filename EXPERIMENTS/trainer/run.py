import os
import pprint
from easydict import EasyDict

from . import exe_dict, parser_dict, run_dict

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
  make_hdf5.run(config, myargs=myargs)
  pass

def ImageNet128_calculate_inception_moments(args, myargs):
  import calculate_inception_moments
  parser = calculate_inception_moments.prepare_parser()
  config = vars(parser.parse_args())
  config = EasyDict(config)

  config1 = myargs.config.ImageNet128_calculate_inception_moments
  for k, v in config1.items():
    setattr(config, k, v)
  config.data_root = os.path.expanduser(config.data_root)
  print(pprint.pformat(config))
  calculate_inception_moments.run(config, myargs)
  pass


def train(args, myargs):
  parser = parser_dict[args.command]()
  config = vars(parser.parse_args())
  config = EasyDict(config)

  config1 = getattr(myargs.config, args.command)
  for k, v in config1.items():
    setattr(config, k, v)
  print(config)
  run_dict[args.command](config, args, myargs)
  pass

def main(args, myargs):
  exe = exe_dict[args.command]
  exec('%s(args, myargs)'%exe)