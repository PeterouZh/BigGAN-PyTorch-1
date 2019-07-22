import utils
import train


exe_dict = {
  'BigGAN_bs256x8': 'train',
  'BigGAN_bs256x8_wgan_gpreal': 'train',
  'BigGAN_bs256x8_wbgan_gpreal': 'train'
}

parser_dict = {
  'BigGAN_bs256x8': utils.prepare_parser,
  'BigGAN_bs256x8_wgan_gpreal': utils.prepare_parser,
  'BigGAN_bs256x8_wbgan_gpreal': utils.prepare_parser
}

run_dict = {
  'BigGAN_bs256x8': train.run,
  'BigGAN_bs256x8_wgan_gpreal': train.run,
  'BigGAN_bs256x8_wbgan_gpreal': train.run
}