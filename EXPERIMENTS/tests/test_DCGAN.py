import numpy as np
import os
import sys
import unittest
import argparse

from template_lib import utils
os.chdir('..')

class Prepare_data(unittest.TestCase):

  def test_Calculate_inception_moments_Celeba64(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6011
        export TIME_STR=1
        export PYTHONPATH=../../submodule:..
        python -c "import test_DCGAN; \
        test_DCGAN.Prepare_data().test_Calculate_inception_moments_Celeba64()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

      # func name
    outdir = os.path.join('results/dataset',
                          sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
                --config DCGAN/configs/dcgan_celeba64.yaml
                --command Celeba64
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

    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from TOOLS import calculate_inception_moments
    calculate_inception_moments.create_inception_moments(args, myargs)
    input('End %s' % outdir)
    return

  def test_Check_inception_moments_Celeba64(self):
    old = np.load(os.path.expanduser(
      '~/ZhouPeng/code/biggan-pytorch/'
      'results/datasets/Celeba_align64_inception_moments.npz'))
    old_mu, old_sigma = old['mu'], old['sigma']
    new = np.load(os.path.expanduser(
      '~/.keras/BigGAN-PyTorch-1/Celeba64_inception_moments.npz'))
    new_mu, new_sigma = new['mu'], new['sigma']
    err_mu, err_sig = np.sum(new_mu - old_mu), np.sum(new_sigma - old_sigma)

    new1 = np.load(os.path.expanduser(
      '~/.keras/BigGAN-PyTorch-1/Celeba64_inception_moments.npz1.npz'))
    new_mu1, new_sigma1 = new1['mu'], new1['sigma']
    err_mu, err_sig = np.sum(new_mu - new_mu1), np.sum(new_sigma - new_sigma1)
    pass


class Testing_Celeba64_DCGAN(unittest.TestCase):

  def test_CelebA64_dcgan_wgan_gp(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=../../submodule:..
        python -c "import test_DCGAN; \
          test_DCGAN.Testing_Celeba64_DCGAN().test_CelebA64_dcgan_wgan_gp()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/DCGAN', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()
    def build_args():
      argv_str = f"""
            --config DCGAN/configs/dcgan_celeba64.yaml
            --command wgan_gp_celeba64
            --resume False 
            --resume_path None 
            --resume_root None 
            --evaluate False --evaluate_path None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str
    args, _ = build_args()
    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from DCGAN.trainer import run
    run.train(args=args, myargs=myargs)
    input('End %s' % outdir)

    return

  def test_CelebA64_dcgan_wbgan_gp(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=1
        export PORT=6007
        export TIME_STR=1
        export PYTHONPATH=../../submodule:..
        python -c "import test_DCGAN; \
          test_DCGAN.Testing_Celeba64_DCGAN().test_CelebA64_dcgan_wbgan_gp()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/DCGAN', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config DCGAN/configs/dcgan_celeba64.yaml
            --command wbgan_gp_celeba64
            --resume False 
            --resume_path None 
            --resume_root None 
            --evaluate False --evaluate_path None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    args.outdir = outdir
    args , myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from DCGAN.trainer import run
    run.train(args=args, myargs=myargs)
    input('End %s' % outdir)

    return

  def test_CelebA64_dcgan_wgan_gpreal(self):
    """
    Usage:
        source activate Peterou_torch_py36
        export CUDA_VISIBLE_DEVICES=1
        export PORT=6007
        export TIME_STR=1
        export PYTHONPATH=../../submodule:..
        python -c "import test_DCGAN; \
          test_DCGAN.Testing_Celeba64_DCGAN().test_CelebA64_dcgan_wgan_gpreal()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/DCGAN', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config DCGAN/configs/dcgan_celeba64.yaml
            --command wgan_gpreal_celeba64
            --resume False 
            --resume_path None 
            --resume_root None 
            --evaluate False --evaluate_path None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, _ = build_args()
    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from DCGAN.trainer import run
    run.train(args=args, myargs=myargs)
    input('End %s' % outdir)

    return

  def test_CelebA64_dcgan_wbgan_gpreal(self):
    """
    Usage:
        source activate Peterou_torch_py36
        export CUDA_VISIBLE_DEVICES=3
        export PORT=6109
        export TIME_STR=1
        export PYTHONPATH=../../submodule:..
        python -c "import test_DCGAN; \
          test_DCGAN.Testing_Celeba64_DCGAN().test_CelebA64_dcgan_wbgan_gpreal()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/DCGAN', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config DCGAN/configs/dcgan_celeba64.yaml
            --command wbgan_gpreal_celeba64
            --resume False 
            --resume_path None 
            --resume_root None 
            --evaluate False --evaluate_path None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, _ = build_args()
    args.outdir = outdir
    args, myargs = utils.config.setup_args_and_myargs(args=args, myargs=myargs)
    from DCGAN.trainer import run
    run.train(args=args, myargs=myargs)
    input('End %s' % outdir)

    return

  def test_CelebA64_dcgan_wbgan_gp_dist(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6111
        export TIME_STR=1
        export PYTHONPATH=../submodule:.
        python -c "import test_DCGAN; \
          test_DCGAN.Testing_Celeba64_DCGAN().test_CelebA64_dcgan_wbgan_gp_dist()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6010'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/DCGAN', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config DCGAN/configs/dcgan_celeba64.yaml
            --command wbgan_gp_dist_celeba64
            --world_size 6
            --resume False 
            --resume_path None 
            --resume_root None 
            --evaluate False --evaluate_path None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from DCGAN.trainer import run
    run.main(args=args, myargs=myargs)
    input('End %s' % outdir)

    return


class test_cifar10_DCGAN(unittest.TestCase):

  def test_cifar10_dcgan_wgan_gp(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=../submodule:.
        python -c "import test_DCGAN; \
          test_DCGAN.test_cifar10_DCGAN().test_cifar10_dcgan_wgan_gp()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/DCGAN', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config DCGAN/configs/dcgan_cifar10.yaml
            --command wgan_gp_cifar10
            --resume False 
            --resume_path None 
            --resume_root None 
            --evaluate False --evaluate_path None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from scripts import dcgan_cifar10
    dcgan_cifar10.run(args=args, myargs=myargs)
    input('End %s' % outdir)
    return

  def test_cifar10_dcgan_wbgan_gp(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=2
        export PORT=6008
        export TIME_STR=1
        export PYTHONPATH=../submodule:.
        python -c "import test_DCGAN; \
          test_DCGAN.test_cifar10_DCGAN().test_cifar10_dcgan_wbgan_gp()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/DCGAN', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config DCGAN/configs/dcgan_cifar10.yaml
            --command wbgan_gp_cifar10
            --resume False 
            --resume_path None 
            --resume_root None 
            --evaluate False --evaluate_path None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str
    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from scripts import dcgan_cifar10
    dcgan_cifar10.run(args=args, myargs=myargs)
    input('End %s' % outdir)
    return

  def test_cifar10_dcgan_wgan_div(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=../submodule:.
        python -c "import test_DCGAN; \
          test_DCGAN.test_cifar10_DCGAN().test_cifar10_dcgan_wgan_div()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/DCGAN', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config DCGAN/configs/dcgan_cifar10.yaml
            --command wgan_div_cifar10
            --resume False 
            --resume_path None 
            --resume_root None 
            --evaluate False --evaluate_path None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from scripts import dcgan_cifar10
    dcgan_cifar10.run(args=args, myargs=myargs)
    input('End %s' % outdir)
    return

  def test_cifar10_dcgan_wbgan_div(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=5
        export PORT=6011
        export TIME_STR=1
        export PYTHONPATH=../submodule:.
        python -c "import test_DCGAN; \
          test_DCGAN.test_cifar10_DCGAN().test_cifar10_dcgan_wbgan_div()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/DCGAN', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config DCGAN/configs/dcgan_cifar10.yaml
            --command wbgan_div_cifar10
            --resume False 
            --resume_path None 
            --resume_root None 
            --evaluate False --evaluate_path None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from scripts import dcgan_cifar10
    dcgan_cifar10.run(args=args, myargs=myargs)
    input('End %s' % outdir)
    return

  def test_cifar10_dcgan_wgan_gpreal(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=2
        export PORT=6008
        export TIME_STR=1
        export PYTHONPATH=../submodule:.
        python -c "import test_DCGAN; \
          test_DCGAN.test_cifar10_DCGAN().test_cifar10_dcgan_wgan_gpreal()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/DCGAN', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config DCGAN/configs/dcgan_cifar10.yaml
            --command wgan_gpreal_cifar10
            --resume False 
            --resume_path None 
            --resume_root None 
            --evaluate False --evaluate_path None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from scripts import dcgan_cifar10
    dcgan_cifar10.run(args=args, myargs=myargs)
    input('End %s' % outdir)
    return

  def test_cifar10_dcgan_wbgan_gpreal(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=4
        export PORT=6010
        export TIME_STR=1
        export PYTHONPATH=../submodule:.
        python -c "import test_DCGAN; \
          test_DCGAN.test_cifar10_DCGAN().test_cifar10_dcgan_wbgan_gpreal()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/DCGAN', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config DCGAN/configs/dcgan_cifar10.yaml
            --command wbgan_gpreal_cifar10
            --resume False 
            --resume_path None 
            --resume_root None 
            --evaluate False --evaluate_path None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from scripts import dcgan_cifar10
    dcgan_cifar10.run(args=args, myargs=myargs)
    input('End %s' % outdir)
    return


class test_cifar10_DCGAN_plot(unittest.TestCase):

  def test_cifar10_dcgan_wgan_gp_shallow(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=2
        export PORT=6108
        export TIME_STR=1
        export PYTHONPATH=../submodule:.
        python -c "import test_DCGAN; \
          test_DCGAN.test_cifar10_DCGAN().test_cifar10_dcgan_wgan_gp()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/DCGAN/plot', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config DCGAN/configs/dcgan_cifar10.yaml
            --command plot_wbgan_gp_shallow
            --resume False 
            --resume_path None 
            --resume_root None 
            --evaluate False --evaluate_path None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from scripts import parse_tensorboard
    parse_tensorboard.parse_tensorboard(args=args, myargs=myargs)
    input('End %s' % outdir)

    return


class Testing_Celeba64_DCGAN_plot(unittest.TestCase):

  def test_plot_large_batchsize(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6111
        export TIME_STR=1
        export PYTHONPATH=../submodule:.
        python -c "import test_DCGAN; \
          test_DCGAN.Testing_Celeba64_DCGAN_plot().test_dcgan_wgan_gp_plot()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/DCGAN/plot', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config DCGAN/configs/dcgan_celeba64.yaml
            --command plot_large_batchsize
            --resume False 
            --resume_path None 
            --resume_root None 
            --evaluate False --evaluate_path None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str
    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from scripts import parse_tensorboard
    parse_tensorboard.parse_tensorboard(args=args, myargs=myargs)
    input('End %s' % outdir)

    return

  def test_plot_bs128(self):
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
        export PORT=6111
        export TIME_STR=1
        export PYTHONPATH=../submodule:.
        python -c "import test_DCGAN; \
          test_DCGAN.Testing_Celeba64_DCGAN_plot().test_plot_bs128()"

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6011'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0'

    # func name
    outdir = os.path.join('results/DCGAN/plot', sys._getframe().f_code.co_name)
    myargs = argparse.Namespace()

    def build_args():
      argv_str = f"""
            --config DCGAN/configs/dcgan_celeba64.yaml
            --command plot_bs128
            --resume False 
            --resume_path None 
            --resume_root None 
            --evaluate False --evaluate_path None
            """
      parser = utils.args_parser.build_parser()
      if len(sys.argv) == 1:
        args = parser.parse_args(args=argv_str.split())
      else:
        args = parser.parse_args()
      args = utils.config_utils.DotDict(vars(args))
      return args, argv_str

    args, argv_str = build_args()

    # parse the config json file
    args = utils.config.process_config(outdir=outdir, config_file=args.config,
                                       resume_root=args.resume_root, args=args,
                                       myargs=myargs)
    from scripts import parse_tensorboard
    parse_tensorboard.parse_tensorboard(args=args, myargs=myargs)
    input('End %s' % outdir)

    return