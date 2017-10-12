# coding=utf-8
# Copyright 2017 The DLT2T Authors.
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

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

# Dependency imports

from DLT2T.data_generators import generator_utils
from DLT2T.data_generators import problem
from DLT2T.data_generators import text_encoder
from DLT2T.utils import registry

import tensorflow as tf

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

@registry.register_problem
class DualLearningEnde(problem.Text2TextProblem):

  @property
  def is_character_level(self):
    return False

  @property
  def vocab_name(self):
    return "vocab.endefr"

  @property
  def targeted_vocab_size(self):
    return 2**15  # 32768

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return problem.SpaceID.DE_TOK

  @property
  def num_shards(self):
    return 1

  @property
  def use_subword_tokenizer(self):
    return True

  def generate_data(self, data_dir, tmp_dir, train_mode, task_id=-1):
    train_paths = self.training_filepaths(
        data_dir, self.num_shards, shuffled=False)
    dev_paths = self.dev_filepaths(
        data_dir, self.num_dev_shards, shuffled=False)
    if self.use_train_shards_for_dev:
      all_paths = train_paths + dev_paths
      generator_utils.generate_files(
          self.generator(data_dir, tmp_dir, True, train_mode), all_paths)
      generator_utils.shuffle_dataset(all_paths)
    else:
      generator_utils.generate_dataset_and_shuffle(
          self.generator(data_dir, tmp_dir, True, train_mode), train_paths,
          self.generator(data_dir, tmp_dir, False, train_mode), dev_paths)

  def generator(self, data_dir, tmp_dir, train, train_mode):
    symbolizer_vocab = generator_utils.get_or_generate_vocab(
        data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size)
    #symbolizer_vocab = text_encoder.SubwordTextEncoder(os.path.join(data_dir, self.vocab_file))
    datasets = _DUAL_ENDE_TRAIN_DATASETS if train else _DUAL_ENDE_TEST_DATASETS
    if train:
      return token_generator(
        train = train,
        train_mode = train_mode,
        A_path = os.path.join(data_dir,datasets[0]),
        B_path = os.path.join(data_dir,datasets[1]),
        A_m_path = os.path.join(data_dir,datasets[2]),
        B_m_path = os.path.join(data_dir,datasets[3]),
        A_hat_path = os.path.join(data_dir,datasets[4]),
        B_hat_path = os.path.join(data_dir,datasets[5]),
        A_score_path = os.path.join(data_dir,datasets[6]),
        B_score_path = os.path.join(data_dir,datasets[7]),
        token_vocab = symbolizer_vocab, 
        eos = EOS)
    else:
      return token_generator(
        train = train,
        train_mode = None,
        A_path = os.path.join(data_dir,datasets[0]),
        B_path = os.path.join(data_dir,datasets[1]),
        token_vocab = symbolizer_vocab,
        eos=EOS)

  def preprocess_examples(self, examples, mode, hparams):
    del mode
    max_seq_length = min(max(hparams.max_input_seq_length,0),max(hparams.max_target_seq_length,0))
    '''
    if hparams.max_input_seq_length > 0:
      examples['A'] = examples['A'][:hparams.max_input_seq_length]
      examples['B_m'] = examples['B_hat'][:hparams.max_input_seq_length]
      examples['A_hat'] = examples['A_hat'][:hparams.max_input_seq_length]
    if hparams.max_target_seq_length > 0:  
      examples['B'] = examples['B'][:hparams.max_target_seq_length]
      examples['B_m'] = examples['B_m'][:hparams.max_target_seq_length]
      examples['A_m'] = examples['A_m'][:hparams.max_input_seq_length]
    '''
    if max_seq_length > 0:
      examples['A'] = examples['A'][:max_seq_length]
      examples['B'] = examples['B'][:max_seq_length]
      examples['A_m'] = examples['A_m'][:max_seq_length]
      examples['B_hat'] = examples['B_hat'][:max_seq_length]
      examples['B_m'] = examples['B_m'][:max_seq_length]
      examples['A_hat'] = examples['A_hat'][:max_seq_length]

    '''
    if hparams.prepend_mode != "none":
      examples["targets"] = tf.concat(
        [examples["inputs"], [0], examples["targets"]], 0)'''
    return examples

  def example_reading_spec(self):
    data_fields = {
        'A': tf.VarLenFeature(tf.int64),
        'B': tf.VarLenFeature(tf.int64),
        'A_m': tf.VarLenFeature(tf.int64),
        'B_hat': tf.VarLenFeature(tf.int64),
        'B_m': tf.VarLenFeature(tf.int64),
        'A_hat': tf.VarLenFeature(tf.int64),
        'A_score': tf.VarLenFeature(tf.float32),
        'B_score': tf.VarLenFeature(tf.float32),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)

def token_generator(
  train, 
  train_mode, 
  A_path, 
  B_path, 
  A_m_path=None, 
  B_m_path=None, 
  A_hat_path=None, 
  B_hat_path=None, 
  A_score_path=None, 
  B_score_path=None, 
  token_vocab=None, 
  eos=None):
  '''
  Refer to token_generator in wmt.py
  Yields:
    A dictionary {"inputs": source-line, "targets": target-line} where
    the lines are integer lists converted from tokens in the file lines.
  '''
  tf.logging.info('Generating tokens...')
  eos_list = [] if eos is None else [eos]
  if not train:
    with tf.gfile.GFile(A_path, mode="r") as A_file:
      with tf.gfile.GFile(B_path, mode="r") as B_file:
        A, B = A_file.readline(), B_file.readline()
        while A and B:
          A_ints = token_vocab.encode(A.strip()) + eos_list
          B_ints = token_vocab.encode(B.strip()) + eos_list
          yield {"A": A_ints, "B": B_ints}
          A, B = A_file.readline(), B_file.readline()
          
  elif train_mode == "pretrain":
    with tf.gfile.GFile(A_path, mode="r") as A_file:
      with tf.gfile.GFile(B_path, mode="r") as B_file:
        A = A_file.readline()
        B = B_file.readline()
        while A and B:
          A_ints = token_vocab.encode(A.strip()) + eos_list
          B_ints = token_vocab.encode(B.strip()) + eos_list
          yield {'A':A_ints, 'B':B_ints}
          A = A_file.readline()
          B = B_file.readline()

  else:
    with tf.gfile.GFile(A_path, mode="r") as A_file:
      with tf.gfile.GFile(B_path, mode="r") as B_file:
        with tf.gfile.GFile(A_m_path, mode="r") as A_m_file:
          with tf.gfile.GFile(B_m_path, mode="r") as B_m_file:
            with tf.gfile.GFile(A_hat_path, mode="r") as A_hat_file:
              with tf.gfile.GFile(B_hat_path, mode="r") as B_hat_file:
                with tf.gfile.GFile(A_score_path, mode="r") as A_score_file:
                  with tf.gfile.GFile(B_score_path, mode="r") as B_score_file:
                    A = A_file.readline()
                    B = B_file.readline()
                    A_m = A_m_file.readline()
                    B_m = B_m_file.readline()
                    A_hat = A_hat_file.readline()
                    B_hat = B_hat_file.readline()
                    A_score = A_score_file.readline()
                    B_score = B_score_file.readline()
                    while A and B and A_m and B_m and A_hat and B_hat and A_score and B_score:
                      A_ints = token_vocab.encode(A.strip()) + eos_list
                      B_ints = token_vocab.encode(B.strip()) + eos_list
                      A_m_ints = token_vocab.encode(A_m.strip()) + eos_list
                      B_m_ints = token_vocab.encode(B_m.strip()) + eos_list
                      A_hat_ints = token_vocab.encode(A_hat.strip()) + eos_list
                      B_hat_ints = token_vocab.encode(B_hat.strip()) + eos_list
                      A_score = [float(A_score.strip())]
                      B_score = [float(B_score.strip())]
                      yield {'A':A_ints, 'B':B_ints, 'A_m':A_m_ints, 'B_m':B_m_ints, 'A_hat':A_hat_ints, 'B_hat':B_hat_ints, 'A_score':A_score, 'B_score':B_score}
                      A = A_file.readline()
                      B = B_file.readline()
                      A_m = A_m_file.readline()
                      B_m = B_m_file.readline()
                      A_hat = A_hat_file.readline()
                      B_hat = B_hat_file.readline()
                      A_score = A_score_file.readline()
                      B_score = B_score_file.readline()
    


_DUAL_ENDE_TRAIN_DATASETS = [
  'parallel_ende.en',
  'parallel_ende.de',
  'mono_ende.en',
  'mono_ende.de',
  'infer_ende.en',
  'infer_ende.de',
  'parallel_ende_score.en',
  'parallel_ende_score.de',
]

_DUAL_ENDE_TEST_DATASETS = [
  'dev_ende.en',
  'dev_ende.de',
]
