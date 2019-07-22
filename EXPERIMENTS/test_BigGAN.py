import os
import sys
import unittest
import argparse

from template_lib import utils


class TestingPrepareData(unittest.TestCase):

  def test_ImageNet128_make_hdf5(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=../submodule:..
        python -c "import test_BigGAN; \
        test_BigGAN.TestingPrepareData().test_ImageNet128_make_hdf5()"
    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/dataset', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/dataset.yaml 
            --command ImageNet128_make_hdf5
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str
    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from trainer import run
    run.main(args, myargs)
    input('End %s' % outdir)
    return

  def test_ImageNet128_calculate_inception_moments(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=../submodule:..
        python -c "import test_BigGAN; \
        test_BigGAN.TestingPrepareData().test_ImageNet128_calculate_inception_moments()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/dataset', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/dataset.yaml 
            --command ImageNet128_calculate_inception_moments
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from trainer import run
    run.main(args, myargs)
    input('End %s' % outdir)
    return


class TestingTrainBigGAN(unittest.TestCase):

  def test_ImageNet128_train(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6011
        export TIME_STR=1
        export PYTHONPATH=../submodule:..
        python -c "import test_BigGAN; \
        test_BigGAN.TestingTrainBigGAN().test_ImageNet128_train()"
    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_imagenet128.yaml 
            --command BigGAN_bs256x8
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str
    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from trainer import run
    run.main(args, myargs)
    input('End %s' % outdir)
    return


class TestingTrainBigGAN_WGAN_GPReal(unittest.TestCase):

  def test_ImageNet128_train_wgan_gpreal(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6011
        export TIME_STR=1
        export PYTHONPATH=../submodule:..
        python -c "import test_BigGAN; \
        test_BigGAN.TestingTrainBigGAN_WGAN_GPReal().test_ImageNet128_train_wgan_gpreal()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_imagenet128.yaml 
            --command BigGAN_bs256x8_wgan_gpreal
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str
    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from trainer import run
    run.main(args, myargs)
    input('End %s' % outdir)
    return

  def test_ImageNet128_train_wbgan_gpreal(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6011
        export TIME_STR=1
        export PYTHONPATH=../submodule:..
        python -c "import test_BigGAN; \
        test_BigGAN.TestingTrainBigGAN_WGAN_GPReal().test_ImageNet128_train_wbgan_gpreal()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/BigGAN', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config ./configs/biggan_imagenet128.yaml 
            --command BigGAN_bs256x8_wbgan_gpreal
            --resume False --resume_path None
            --resume_root None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args.CUDA_VISIBLE_DEVICES = os.environ['CUDA_VISIBLE_DEVICES']
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from trainer import run
    run.main(args, myargs)
    input('End %s' % outdir)
    return


