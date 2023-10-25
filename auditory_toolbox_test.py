import sys
from collections import Counter

from absl.testing import absltest
import numpy as np

import auditory_toolbox as pat


class ClusterTests(absltest.TestCase):
  def test_erb_space(self):
    low_freq = 100.0
    high_freq = 44100/4.0
    num_channels = 100
    cfArray = pat.ERBSpace(lowFreq = low_freq, highFreq = high_freq, 
                           n = num_channels)
    self.assertLen(cfArray, num_channels)
    self.assertAlmostEqual(cfArray[-1], low_freq)
    self.assertLess(cfArray[0], high_freq)


if __name__=="__main__": 
  absltest.main()
