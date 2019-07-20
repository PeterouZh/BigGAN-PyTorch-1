import utils
import train


exe_dict = {
  'BigGAN_bs256x8': 'train'
}

parser_dict = {
  'BigGAN_bs256x8': utils.prepare_parser
}

run_dict = {
  'BigGAN_bs256x8': train.run
}