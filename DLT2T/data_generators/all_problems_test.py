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

"""Tests for DLT2T's all_problems.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from DLT2T.data_generators import all_problems

import tensorflow as tf


class AllProblemsTest(tf.test.TestCase):

  def testImport(self):
    """Make sure that importing all_problems doesn't break."""
    self.assertIsNotNone(all_problems)


if __name__ == '__main__':
  tf.test.main()
