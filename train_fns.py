''' train_fns.py
Functions for the main loop of training different conditional image models
'''
import collections
import random

import torch
import torch.nn as nn
import torchvision
import os

import utils
import losses

import numpy as np

from template_lib.v2.config import global_cfg
from exp.multilabel_ce import loss_multilabel_ce


# Dummy training function for debugging
def dummy_training_function():
  def train(x, y):
    return {}
  return train


def GAN_training_function(G, D, GD, z_, y_, ema, state_dict, config):
  dd = collections.defaultdict(dict)
  def train(x, y):
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    counter = 0
    
    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)
    
    
    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      # The fake class label
      lossy = torch.LongTensor(config['batch_size'])
      lossy = lossy.cuda()
      lossy.data.fill_(config['n_classes']) # index for fake just for loss
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()

        D_fake, D_real = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                                        x[counter], y[counter], train_G=False,
                                        split_D=config['split_D'])
        
        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations
        if getattr(global_cfg, 'multilabel_ce', False):
          default_v = getattr(global_cfg, 'default_v', -1)
          # rand_num = random.uniform(0, 1)

          D_real_positive = [y[counter], config['n_classes']]
          D_real_negative = (config['n_classes'] + 1, )
          D_loss_real = loss_multilabel_ce.discriminator_loss(
            pred=D_real, positive=D_real_positive, negative=D_real_negative, default_v=default_v)

          D_fake_positive = (config['n_classes'] + 1, )
          D_fake_negative = (y_[:config['batch_size']], config['n_classes'])
          D_loss_fake = loss_multilabel_ce.discriminator_loss(
            pred=D_fake, positive=D_fake_positive, negative=D_fake_negative,default_v=default_v)
          # D_loss_fake = loss_multilabel_ce.discriminator_loss(
          #   D_fake, y_[:config['batch_size']], config['n_classes'] + 1)

        elif config['mh_csc_loss'] or config['mh_loss']:
          D_loss_real = losses.crammer_singer_criterion(D_real, y[counter])
          D_loss_fake = losses.crammer_singer_criterion(D_fake, lossy[:config['batch_size']])
        else:
          D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) \
                 / float(config['num_D_accumulations'])
        D_loss.backward()
        counter += 1
        
      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])
      
      D.optim.step()
    
    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, False)
      utils.toggle_grad(G, True)
      
    # Zero G's gradients by default before training G, for safety
    G.optim.zero_grad()
    
    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']):    
      # reusing the same noise for CIFAR ...
      if config['resampling'] or (accumulation_index > 0):
        z_.sample_()
        y_.sample_()
      
      if config['fm_loss']:
        D_feat_fake, D_feat_real = GD(z_, y_, x[-1], None, train_G=True, split_D=config['split_D'], feat=True)
        fm_loss = torch.mean(torch.abs(torch.mean(D_feat_fake, 0) - torch.mean(D_feat_real, 0)))
        G_loss = fm_loss
      else:
        D_fake = GD(z_, y_, train_G=True, split_D=config['split_D'])
        if getattr(global_cfg, 'multilabel_ce', False):
          G_loss = 0
          if global_cfg.weight_adv > 0.:
            G_fake_positive = (y_, config['n_classes'])
            G_fake_negative = (config['n_classes'] + 1, )
            adv_loss = loss_multilabel_ce.discriminator_loss(
              pred=D_fake, positive=G_fake_positive, negative=G_fake_negative, default_v=default_v)
            G_loss = G_loss + global_cfg.weight_adv * adv_loss
          if global_cfg.weight_fm > 0.:
            D_feat_fake, D_feat_real = GD(z_, y_, x[-1], None, train_G=True, split_D=config['split_D'], feat=True)
            fm_loss = torch.mean(torch.abs(torch.mean(D_feat_fake, 0) - torch.mean(D_feat_real, 0)))
            G_loss = G_loss + global_cfg.weight_fm * fm_loss

          G_loss = G_loss / float(config['num_G_accumulations'])
        elif config['mh_csc_loss']:
          G_loss = losses.crammer_singer_complement_criterion(D_fake, lossy[:config['batch_size']]) / float(config['num_G_accumulations'])
        elif config['mh_loss']:
          D_feat_fake, D_feat_real = GD(z_, y_, x[-1], None, train_G=True, split_D=config['split_D'], feat=True)
          fm_loss = torch.mean(torch.abs(torch.mean(D_feat_fake, 0) - torch.mean(D_feat_real, 0)))
          oth_loss = losses.mh_loss(D_fake, y_[:config['batch_size']])
          G_loss = (config['mh_fmloss_weight']*fm_loss + config['mh_loss_weight']*oth_loss) / float(config['num_G_accumulations'])
        else:
          G_loss = losses.generator_loss(D_fake) / float(config['num_G_accumulations'])
      G_loss.backward()
    
    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'], 
                  blacklist=[param for param in G.shared.parameters()])
    G.optim.step()
    
    # If we have an ema, update it, regardless of if we test with it or not
    if config['ema']:
      ema.update(state_dict['itr'])

    dd.clear()
    dd['D_loss'].update({'G_loss': float(G_loss.item()),
                         'D_loss_real': float(D_loss_real.item()),
                         'D_loss_fake': float(D_loss_fake.item())})
    # Return G's loss and the components of D's loss.
    return dd
  return train

def GAN_training_function_with_unlabeled(G, D, GD, z_, y_, ema, state_dict, config):
  def train(x, y, x_unlab):
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    x_unlab = torch.split(x_unlab, config['batch_size'])
    counter = 0
    
    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)
    
    
    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      lossy = torch.LongTensor(config['batch_size'])
      lossy = lossy.cuda()
      lossy.data.fill_(config['n_classes']) # index for fake just for loss
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()        

        D_fake, D_real = GD(z_[:config['batch_size']], y_[:config['batch_size']], 
                            x[counter], y[counter], train_G=False, 
                            split_D=config['split_D'])
        D_unlab = D(x_unlab[counter])
        
        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations
        if config['mh_csc_loss'] or config['mh_loss']:
          D_loss_real = losses.crammer_singer_criterion(D_real, y[counter])
          D_loss_fake = losses.crammer_singer_criterion(D_fake, lossy[:config['batch_size']])
          D_loss_real += losses.crammer_singer_complement_criterion(D_unlab, lossy[:config['batch_size']])
          D_loss_real /= 2.0
        else:
          D_loss_real, D_loss_fake = losses.discriminator_loss(D_fake, torch.cat([D_real, D_unlab]))
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
        D_loss.backward()
        counter += 1
        
      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])
      
      D.optim.step()
    
    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, False)
      utils.toggle_grad(G, True)
      
    # Zero G's gradients by default before training G, for safety
    G.optim.zero_grad()
    
    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']):    
      # reusing the same noise for CIFAR ...
      if config['resampling'] or (accumulation_index > 0):
        z_.sample_()
        y_.sample_()

      if config['fm_loss']:
        D_feat_fake, D_feat_unlab = GD(z_, y_, x_unlab[-1], None, train_G=True, split_D=config['split_D'], feat=True)
        fm_loss = torch.mean(torch.abs(torch.mean(D_feat_fake, 0) - torch.mean(D_feat_unlab, 0)))
        G_loss = fm_loss
      else:
        D_fake = GD(z_, y_, train_G=True, split_D=config['split_D'])
        if config['mh_csc_loss']:
          G_loss = losses.crammer_singer_complement_criterion(D_fake, lossy[:config['batch_size']]) / float(config['num_G_accumulations'])
        elif config['mh_loss']:
          D_feat_fake, D_feat_unlab = GD(z_, y_, x_unlab[-1], None, train_G=True, split_D=config['split_D'], feat=True)
          fm_loss = torch.mean(torch.abs(torch.mean(D_feat_fake, 0) - torch.mean(D_feat_unlab, 0)))
          oth_loss = losses.mh_loss(D_fake, y_[:config['batch_size']])
          G_loss = (config['mh_fmloss_weight']*fm_loss + config['mh_loss_weight']*oth_loss) / float(config['num_G_accumulations'])
        else:
          G_loss = losses.generator_loss(D_fake) / float(config['num_G_accumulations'])
        

      G_loss.backward()
    
    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'], 
                  blacklist=[param for param in G.shared.parameters()])
    G.optim.step()
    
    # If we have an ema, update it, regardless of if we test with it or not
    if config['ema']:
      ema.update(state_dict['itr'])
    
    out = {'G_loss': float(G_loss.item()), 
            'D_loss_real': float(D_loss_real.item()),
            'D_loss_fake': float(D_loss_fake.item())}
    # Return G's loss and the components of D's loss.
    return out
  return train
  
''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets. '''
def save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, 
                    state_dict, config, experiment_name):
  utils.save_weights(G, D, state_dict, config['weights_root'],
                     experiment_name, None, G_ema if config['ema'] else None)
  # Save an additional copy to mitigate accidental corruption if process
  # is killed during a save (it's happened to me before -.-)
  if config['num_save_copies'] > 0:
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name,
                       'copy%d' %  state_dict['save_num'],
                       G_ema if config['ema'] else None)
    state_dict['save_num'] = (state_dict['save_num'] + 1 ) % config['num_save_copies']
    
  # Use EMA G for samples or non-EMA?
  which_G = G_ema if config['ema'] and config['use_ema'] else G
  
  # Accumulate standing statistics?
  if config['accumulate_stats']:
    utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                           z_, y_, config['n_classes'],
                           config['num_standing_accumulations'])
  
  # Save a random sample sheet with fixed z and y      
  with torch.no_grad():
    if config['parallel']:
      fixed_Gz =  nn.parallel.data_parallel(which_G, (fixed_z, which_G.shared(fixed_y)))
    else:
      fixed_Gz = which_G(fixed_z, which_G.shared(fixed_y))
  if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
    os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
  image_filename = '%s/%s/fixed_samples%010d.jpg' % (config['samples_root'],
                                                  experiment_name,
                                                  state_dict['itr'])
  torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                             nrow=int(fixed_Gz.shape[0] **0.5), normalize=True)
  # For now, every time we save, also save sample sheets
  utils.sample_sheet(which_G,
                     classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']],
                     num_classes=config['n_classes'],
                     samples_per_class=10, parallel=config['parallel'],
                     samples_root=config['samples_root'],
                     experiment_name=experiment_name,
                     folder_number=state_dict['itr'],
                     z_=z_)
  # Also save interp sheets
  for fix_z, fix_y in zip([False, False, True], [False, True, False]):
    utils.interp_sheet(which_G,
                       num_per_sheet=16,
                       num_midpoints=8,
                       num_classes=config['n_classes'],
                       parallel=config['parallel'],
                       samples_root=config['samples_root'],
                       experiment_name=experiment_name,
                       folder_number=state_dict['itr'],
                       sheet_number=0,
                       fix_z=fix_z, fix_y=fix_y, device='cuda')


  
''' This function runs the inception metrics code, checks if the results
    are an improvement over the previous best (either in IS or FID, 
    user-specified), logs the results, and saves a best_ copy if it's an 
    improvement. '''
def test(G, D, G_ema, z_, y_, state_dict, config, sample, get_inception_metrics,
         experiment_name, test_log):
  print('Gathering inception metrics...')
  if config['accumulate_stats']:
    utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                           z_, y_, config['n_classes'],
                           config['num_standing_accumulations'])
  IS_mean, IS_std, FID = get_inception_metrics(
    sample, eval_iter=state_dict['epoch'],
    num_inception_images=config['num_inception_images'], num_splits=10)
  print('Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (state_dict['itr'], IS_mean, IS_std, FID))
  # If improved over previous best metric, save approrpiate copy
  if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
    or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])):
    print('%s improved over previous best, saving checkpoint...' % config['which_best'])
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name, 'best%d' % state_dict['save_best_num'],
                       G_ema if config['ema'] else None)
    state_dict['save_best_num'] = (state_dict['save_best_num'] + 1 ) % config['num_best_copies']
  state_dict['best_IS'] = max(state_dict['best_IS'], IS_mean)
  state_dict['best_FID'] = min(state_dict['best_FID'], FID)
  # Log results to file
  test_log.log(itr=int(state_dict['itr']), IS_mean=float(IS_mean),
               IS_std=float(IS_std), FID=float(FID))
               
def classify(G, D, G_ema, z_, y_, state_dict, config, sample, get_inception_metrics,
         experiment_name, test_log):
  # TODO     
  pass