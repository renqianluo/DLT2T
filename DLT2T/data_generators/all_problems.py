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

"""Imports for problem modules."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from DLT2T.data_generators import algorithmic
from DLT2T.data_generators import algorithmic_math
from DLT2T.data_generators import audio
from DLT2T.data_generators import cipher
from DLT2T.data_generators import cnn_dailymail
from DLT2T.data_generators import desc2code
from DLT2T.data_generators import ice_parsing
from DLT2T.data_generators import image
from DLT2T.data_generators import imdb
from DLT2T.data_generators import lm1b
from DLT2T.data_generators import ptb
from DLT2T.data_generators import snli
from DLT2T.data_generators import wiki
from DLT2T.data_generators import wmt
from DLT2T.data_generators import wsj_parsing


# Problem modules that require optional dependencies
# pylint: disable=g-import-not-at-top
try:
  # Requires h5py
  from DLT2T.data_generators import gene_expression
except ImportError:
  pass
# pylint: enable=g-import-not-at-top
# pylint: enable=unused-import
