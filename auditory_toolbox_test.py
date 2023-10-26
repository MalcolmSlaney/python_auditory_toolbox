import sys
from collections import Counter

from absl.testing import absltest
import numpy as np
from scipy import signal

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

  def test_make_erb_filters(self):
    # Ten channel ERB Filterbank.  Make sure return has the right size.
    # Will test coefficients when we test the filterbank.
    fs = 16000
    low_freq = 100
    num_chan = 10
    fcoefs = pat.MakeERBFilters(fs, num_chan, low_freq)
    self.assertLen(fcoefs, 10)

    A0, A11, A12, A13, A14, A2, B0, B1, B2, gain = fcoefs
    self.assertEqual(A0.shape, (num_chan,))
    self.assertEqual(A11.shape, (num_chan,))
    self.assertEqual(A12.shape, (num_chan,))
    self.assertEqual(A13.shape, (num_chan,))
    self.assertEqual(A14.shape, (num_chan,))
    self.assertEqual(B0.shape, (num_chan,))
    self.assertEqual(B1.shape, (num_chan,))
    self.assertEqual(gain.shape, (num_chan,))

  def test_erb_filterbank(self):
    fs = 16000
    low_freq = 100
    num_chan = 10
    fcoefs = pat.MakeERBFilters(fs, num_chan, low_freq)

    impulse_len = 512
    x = np.zeros(impulse_len)
    x[0] = 1

    y = pat.ERBFilterBank(x, fcoefs)
    self.assertEqual(y.shape, (num_chan, impulse_len))

    resp = 20*np.log10(np.abs(np.fft.fft(y.T, axis=0)))

    # Test to make sure spectral peaks are in the right place for each channel
    matlab_peak_locs = np.array([184, 132, 94, 66, 46, 32, 21, 14, 8, 4])
    python_peak_locs = np.argmax(resp[:impulse_len//2], axis=0)

    # Add one to python locs because Matlab arrays start at 1
    np.testing.assert_equal(matlab_peak_locs, python_peak_locs+1)

  def test_correlogram_array(self):
    test_impulses = np.zeros((1,1024))
    test_impulses[0, range(0, test_impulses.shape[1], 100)] = 1
    test_frame = pat.CorrelogramFrame(test_impulses, 256)
    np.testing.assert_equal(np.where(test_frame > 0.1)[1], [0, 100, 200])

    # Now test with cochlear input to correlogram
    impulse_len = 512
    fs = 16000
    low_freq = 100
    num_chan = 64
    fcoefs = pat.MakeERBFilters(fs, num_chan, low_freq)

    # Make harmonic input signal
    s = 0
    for h in range(1, 10):
      s = s + np.sin(2*np.pi*np.arange(impulse_len)/200*h)

    y = pat.ERBFilterBank(s, fcoefs)
    frame_width = 256
    frame = pat.CorrelogramFrame(y, frame_width)
    self.assertEqual(frame.shape, (num_chan, frame_width))

    # Make sure the top channels have no output.
    no_output = np.where(np.sum(frame, 1) < 0.2)
    np.testing.assert_equal(no_output[0], np.arange(36))

  def test_mfcc(self):
    sample_rate = 16000.0
    f0 = 2000
    tone = np.sin(2*np.pi*f0*np.arange(4000)/sample_rate)
    [ceps,freqresp,fb,fbrecon,freqrecon]= pat.mfcc(tone,sample_rate,100);

    fftSize = 512  # From the MFCC source code
    freqs = np.arange(fftSize)*sample_rate/fftSize
    self.assertEqual(f0/sample_rate*fftSize, np.argmax(np.sum(freqrecon, axis=1)))

if __name__=="__main__": 
  absltest.main()
