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
"""Training NCSN++ on Church with VE SDE."""

from configs.default_ve_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = True

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'

  # data
  data = config.data
  data.dataset = 'CelebAHQ'
  data.image_size = 128
  #TRAIN
  #FFHQ
  #data.tfrecords_path = PATH TO FFHQ TFRECORDS  
  
  #TEST
  #data.tfrecords_path = PATH TO TEST DATASET

  ## SAMPLE CELEBA
  #data.tfrecords_path = './sample_imgs/tfrecords/tfrecords-r07.tfrecords'

  ## celeba 500 ids
  #data.tfrecords_path = '/home/msantos/sr-sde/experimentos_celeba/prova_HR_tfrecords/prova_HR_tfrecords-r07.tfrecords'

  #data.tfrecords_path = '/home/msantos/quis-campi-completo-aj/tf_01/tf_01-r07.tfrecords'
  #data.tfrecords_path = '/home/msantos/quis-campi-completo-aj/tf_02/tf_02-r07.tfrecords'
  #data.tfrecords_path = '/home/msantos/quis-campi-completo-aj/tf_03/tf_03-r07.tfrecords'
  data.tfrecords_path = '/home/msantos/quis-campi-completo-aj/tf_04/tf_04-r07.tfrecords'
  #data.tfrecords_path = '/home/msantos/quis-campi-completo-aj/tf_05/tf_05-r07.tfrecords'

  ## FEATURES CELEBA 
  #data.features_path = './sample_imgs/features/sample_features_celeba.t'
  #MEAN
  #data.features_path = '/home/msantos/sr-sde/experimentos_celeba/media_features_500ids/celeba_500ids_feature_ir18_media.t'
  # NN
  #data.features_path = '/home/msantos/AdaFace/feat_sibgrapi/combined_prova500ids_dropout_corrigido.pt'
  #data.features_path = '/home/msantos/AdaFace/feat_sibgrapi/combined_prova500ids_5feat.pt'
  #data.features_path = '/home/msantos/AdaFace/feat_sibgrapi/combined_prova500ids_16feat.pt'
  ## rede delta treinada no datasetcasia
  #data.features_path = '/home/msantos/AdaFace/feat_sibgrapi/combined_prova500ids_traincasia.pt' ## treinado no casia

  ## FEATURES QUIS
  # MEAN
  #data.features_path = '/home/msantos/sr-sde/1-IP-input-hr/features_media_LR_ir_18/quiscampi_feature_media.t'
  # NN
  #data.features_path = '/home/msantos/AdaFace/feat_sibgrapi/combined_NN_quis.t'
  #data.features_path = '/home/msantos/AdaFace/feat_sibgrapi/combined_NN_quis_dropout_corrigido.t'
  # rede delta treinada no dataset casia
  data.features_path = '/home/msantos/AdaFace/feat_sibgrapi/combined_NN_quis_delta_treinado_casia.t'

  ## ADAFACE PATH
  data.adaface_model_path = './pretrained_adaface/adaface_ir18_casia.ckpt'

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.sigma_max = 348
  model.sigma_min = 0.001
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  #model.progressive = 'output_skip'
  model.progressive = 'none'
  model.progressive_input = 'input_skip'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  return config
