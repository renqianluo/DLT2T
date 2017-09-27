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

  def generator(self, data_dir, tmp_dir, train):
    #symbolizer_vocab = generator_utils.get_or_generate_vocab(
    #    data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size)
    symbolizer_vocab = text_encoder.SubwordTextEncoder(os.path.join(data_dir, vocab_filename))
    datasets = _ENDE_TRAIN_DATASETS if train else _ENDE_TEST_DATASETS
    if train:
      return token_generator(os.path.join(datadir,datasets[0]), os.path.join(datadir,datasets[1]), 
          os.path.join(datadir,datasets[2]), os.path.join(datadir,datasets[3]), symbolizer_vocab, EOS)
    else:
      return token_generator(os.path.join(datadir,datasets[0]), os.path.join(datadir,datasets[1]), 
          token_vocab=symbolizer_vocab, eos=EOS)

  def example_reading_spec(self):
    data_fields = {
        "A": tf.VarLenFeature(tf.int64),
        "B": tf.VarLenFeature(tf.int64),
        "A_m": tf.VarLenFeature(tf.int64),
        "B_m": tf.VarLenFeature(tf.int64)
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)

def token_generator(A_path, B_path, A_m_path=None, B_m_path=None, token_vocab=None, eos=None):
  '''
  Refer to token_generator in wmt.py
  Yields:
    A dictionary {"inputs": source-line, "targets": target-line} where
    the lines are integer lists converted from tokens in the file lines.
  '''
  eos_list = [] if eos is None else [eos]
  if A_m is None and B_m is None:  
    with tf.gfile.GFile(A, mode="r") as A_file:
      with tf.gfile.GFile(B, mode="r") as B_file:
        A, B = A_file.readline(), B_file.readline()
        while A and B:
          A_ints = token_vocab.encode(A.strip()) + eos_list
          B_ints = token_vocab.encode(B.strip()) + eos_list
          yield {"A": A_ints, "B": B_ints}
          A, B = A_file.readline(), B_file.readline()
  else:
    with tf.gfile.GFile(A_path, mode="r") as A_file:
      with tf.gfile.GFile(B_path, mode="r") as B_file:
        with tf.gfile.GFile(A_m_path, mode="r") as A_m_file:
          with tf.gfile.GFile(B_m_path, mode="r") as B_m_file:
            A, B, A_m, B_m = A_file.readline(), B_file.readline(), A_m_file.readline(), B_m_file.readline()
            while A and B and A_m and B_m:
              score_A = A.strip().rfind(' ')
              score_B = B.strip().rfind(' ')
              score_A_m = A_m.strip().rfind(' ')
              score_B_m = A_m.strip().rfind(' ')
              A_ints = token_vocab.encode(A.strip()[:score_A]) + eos_list + [float(A.strip()[score_A:])]
              B_ints = token_vocab.encode(B.strip()[:score_B]) + eos_list + [float(B.strip()[score_B:])]
              A_m_ints = token_vocab.encode(A_m.strip()[:score_A_m]) + eos_list + [float(A_m.strip()[score_A_m:])*1000000+1000000]
              B_m_ints = token_vocab.encode(B_m.strip()[:score_B_m]) + eos_list + [float(B_m.strip()[score_B_m:])*1000000+1000000]
              yield {"A": A_ints, "B": B_ints, "A_m":A_m_ints, "B_m":B_m_ints}
              A, B, A_m, B_m = A_file.readline(), B_file.readline(), A_m_file.readline(), B_m_file.readline()
    


_DUAL_ENDE_TRAIN_DATASETS = [
  'dual_ende.en',
  'dual_ende.de',
  'dual_en',
  'dual_de'
]

_DUAL_ENDE_TRAIN_DATASETS = [
  'dual_ende.en',
  'dual_ende.de',
]
