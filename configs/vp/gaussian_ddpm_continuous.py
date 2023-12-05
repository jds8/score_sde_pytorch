# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training DDPM with sub-VP SDE."""

from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.batch_size = 512
  training.sde = 'vpsde'
  training.continuous = True
  training.reduce_mean = True
  training.restore_checkpoint = False
  training.mu = 0.
  training.sigma = 1.
  training.snapshot_freq = 500
  training.snapshot_sampling = False
  training.likelihood_weighting = False

  evaluate = config.eval
  evaluate.begin_ckpt = 1
  evaluate.end_ckpt = 1
  evaluate.enable_sampling = True
  evaluate.num_samples = 100
  evaluate.enable_loss = False
  evaluate.enable_bpd = False
  evaluate.use_bpd = False

  # sampling
  sampling = config.sampling
  sampling.method = 'ode'
  sampling.predictor = 'none'
  sampling.corrector = 'none'
  sampling.noise_removal = False
  sampling.probability_flow = True

  # data
  data = config.data
  data.centered = True
  data.num_channels = 1

  # model
  model = config.model
  model.name = 'nnet'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 1
  model.ch_mult = (1, 1, 1, 1)
  model.num_res_blocks = 1
  model.attn_resolutions = (1,)
  model.resamp_with_conv = True
  model.conditional = True
  model.kernel_size = 1
  model.padding = 0
  model.num_groups = 1
  model.up_mult = 1
  model.d_model = 1
  model.cond_dim = 1

  return config