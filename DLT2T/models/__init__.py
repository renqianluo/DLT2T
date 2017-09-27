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

"""Models defined in T2T. Imports here force registration."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

# pylint: disable=unused-import

from DLT2T.layers import modalities
from DLT2T.models import attention_lm
from DLT2T.models import attention_lm_moe
from DLT2T.models import bluenet
from DLT2T.models import bytenet
from DLT2T.models import cycle_gan
from DLT2T.models import gene_expression
from DLT2T.models import lstm
from DLT2T.models import multimodel
from DLT2T.models import neural_gpu
from DLT2T.models import shake_shake
from DLT2T.models import slicenet
from DLT2T.models import transformer
from DLT2T.models import transformer_alternative
from DLT2T.models import transformer_moe
from DLT2T.models import transformer_revnet
from DLT2T.models import transformer_vae
from DLT2T.models import xception
# pylint: enable=unused-import
