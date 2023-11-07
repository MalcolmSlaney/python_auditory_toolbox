"""Code to test the auditory toolbox."""
from absl.testing import absltest
import numpy as np
import scipy

import auditory_toolbox as pat


class AuditoryToolboxTests(absltest.TestCase):
  """Test cases for auditory toolbox."""
  def test_erb_space(self):
    low_freq = 100.0
    high_freq = 44100/4.0
    num_channels = 100
    cf_array = pat.ErbSpace(low_freq = low_freq, high_freq = high_freq,
                           n = num_channels)
    self.assertLen(cf_array, num_channels)
    # Make sure low and high CF's are where we expect them to be.
    self.assertAlmostEqual(cf_array[-1], low_freq)
    self.assertLess(cf_array[0], high_freq)

  def test_make_erb_filters(self):
    # Ten channel ERB Filterbank.  Make sure return has the right size.
    # Will test coefficients when we test the filterbank.
    fs = 16000
    low_freq = 100
    num_chan = 10
    fcoefs = pat.MakeErbFilters(fs, num_chan, low_freq)
    self.assertLen(fcoefs, 10)

    # Test all the filter coefficient array shapes
    a0, a11, a12, a13, a14, a2, b0, b1, b2, gain = fcoefs
    self.assertEqual(a0.shape, (num_chan,))
    self.assertEqual(a11.shape, (num_chan,))
    self.assertEqual(a12.shape, (num_chan,))
    self.assertEqual(a13.shape, (num_chan,))
    self.assertEqual(a14.shape, (num_chan,))
    self.assertEqual(a2.shape, (num_chan,))
    self.assertEqual(b0.shape, (num_chan,))
    self.assertEqual(b1.shape, (num_chan,))
    self.assertEqual(b2.shape, (num_chan,))
    self.assertEqual(gain.shape, (num_chan,))

  def test_erb_filterbank(self):
    fs = 16000
    low_freq = 100
    num_chan = 10
    fcoefs = pat.MakeErbFilters(fs, num_chan, low_freq)

    impulse_len = 512
    x = np.zeros(impulse_len)
    x[0] = 1

    y = pat.ErbFilterBank(x, fcoefs)
    self.assertEqual(y.shape, (num_chan, impulse_len))
    self.assertAlmostEqual(np.max(y), 0.10657410, delta=0.01)

    resp = 20*np.log10(np.abs(np.fft.fft(y.T, axis=0)))

    # Test to make sure spectral peaks are in the right place for each channel
    matlab_peak_locs = np.array([184, 132, 94, 66, 46, 32, 21, 14, 8, 4])
    python_peak_locs = np.argmax(resp[:impulse_len//2], axis=0)

    # Add one to python locs because Matlab arrays start at 1
    np.testing.assert_equal(matlab_peak_locs, python_peak_locs+1)

  def test_correlogram_array(self):
    def local_peaks(x):
      i = np.argwhere(np.logical_and(x[:-2] < x[1:-1],
                                    x[2:] < x[1:-1])) + 1
      return [j[0] for j in i]

    test_impulses = np.zeros((1,1024))
    test_impulses[0, range(0, test_impulses.shape[1], 100)] = 1
    test_frame = pat.CorrelogramFrame(test_impulses, 256)
    np.testing.assert_equal(np.where(test_frame > 0.1)[1], [0, 100, 200])

    # Now test with cochlear input to correlogram
    impulse_len = 512
    fs = 16000
    low_freq = 100
    num_chan = 64
    fcoefs = pat.MakeErbFilters(fs, num_chan, low_freq)

    # Make harmonic input signal
    s = 0
    pitch_lag = 200
    for h in range(1, 10):
      s = s + np.sin(2*np.pi*np.arange(impulse_len)/pitch_lag*h)

    y = pat.ErbFilterBank(s, fcoefs)
    frame_width = 256
    frame = pat.CorrelogramFrame(y, frame_width)
    self.assertEqual(frame.shape, (num_chan, frame_width))
    self.assertGreaterEqual(np.min(frame), 0.0)

    # Make sure the top channels have no output.
    spectral_profile = np.sum(frame, 1)
    no_output = np.where(spectral_profile < 2)
    np.testing.assert_equal(no_output[0], np.arange(31))

    # Make sure we have spectral peaks at the right locations
    spectral_peaks = local_peaks(spectral_profile)
    self.assertEqual(spectral_peaks, [42, 44, 46, 48, 50, 53, 56, 60])

    # Make sure the first peak (after 0 lag) is at the pitch lag
    summary_correlogram = np.sum(frame, 0)
    skip_lags = 100
    self.assertEqual(np.argmax(summary_correlogram[skip_lags:]) + skip_lags,
                     pitch_lag)

  def test_correlogram_pitch(self):
    sample_len = 20000
    sample_rate = 22254
    pitch_center = 120
    u = pat.MakeVowel(sample_len, pat.FMPoints(sample_len, pitch_center),
                      sample_rate, 'u')

    low_freq = 60
    num_chan = 100
    fcoefs = pat.MakeErbFilters(sample_rate, num_chan, low_freq)
    coch = pat.ErbFilterBank(u, fcoefs)
    cor = pat.CorrelogramArray(coch,sample_rate,50,256)
    [pitch,sal] = pat.CorrelogramPitch(cor, 256, sample_rate,100,200)

    # Make sure center and overall pitch deviation are as expected.
    self.assertAlmostEqual(np.mean(pitch), pitch_center, delta=2)
    self.assertAlmostEqual(np.min(pitch), pitch_center-6, delta=2)
    self.assertAlmostEqual(np.max(pitch), pitch_center+6, delta=2)
    np.testing.assert_array_less(0.8, sal[:40])

    # Now test salience when we add noise
    n = np.random.randn(sample_len) * np.arange(sample_len)/sample_len
    un=u + n/4

    low_freq = 60
    num_chan = 100
    fcoefs = pat.MakeErbFilters(sample_rate, num_chan, low_freq)
    coch= pat.ErbFilterBank(un, fcoefs)
    cor = pat.CorrelogramArray(coch,sample_rate,50,256)
    [pitch,sal] = pat.CorrelogramPitch(cor,256,22254,100,200)

    lr = scipy.stats.linregress(range(len(sal)), y=sal, alternative='less')
    self.assertAlmostEqual(lr.slope, -0.012, delta=0.001)
    self.assertAlmostEqual(lr.rvalue, -0.963, delta=0.02)

  def test_mfcc(self):
    # Put a tone into MFCC and make sure it's in the right
    # spot in the reconstruction.
    sample_rate = 16000.0
    f0 = 2000
    tone = np.sin(2*np.pi*f0*np.arange(4000)/sample_rate)
    [_,_,_,_,freqrecon]= pat.Mfcc(tone,sample_rate,100)

    fft_size = 512  # From the MFCC source code
    self.assertEqual(f0/sample_rate*fft_size,
                     np.argmax(np.sum(freqrecon, axis=1)))

  def test_fm_points  (self):
    base_pitch = 160
    sample_rate = 16000
    fmfreq = 10
    fmamp = 20
    points = pat.FMPoints(100000, base_pitch, fmfreq, fmamp, 16000)

    # Make sure the average glottal pulse locations is 1 over the pitch
    d_points = points[1:] - points[:-1]
    self.assertAlmostEqual(np.mean(d_points), sample_rate/base_pitch, delta=1)

    # Make sure the frequency deviation is as expected.
    # ToDo(malcolm): Test the deviation, it's not right!

  def test_make_vowel(self):
    def local_peaks(x):
      i = np.argwhere(np.logical_and(x[:-2] < x[1:-1],
                                    x[2:] < x[1:-1])) + 1
      return [j[0] for j in i]

    test_seq = local_peaks(np.array([1,2,3,2,1,1,2,2,3,4,1]))
    np.testing.assert_equal(test_seq, np.array([2, 9]))

    def vowel_peaks(vowel):
      """Synthesize a vowel and find the frequencies of the spectral peaks"""
      sample_rate = 16000
      vowel = pat.MakeVowel(1024, [1,], sample_rate, vowel)
      spectrum = 20*np.log10(np.abs(np.fft.fft(vowel)))
      freqs = np.arange(len(vowel))*sample_rate/len(vowel)
      return freqs[local_peaks(spectrum)[:3]]

    # Make sure the spectrum of each vowel has peaks in the right spots.
    bin_width = 16000/1024
    np.testing.assert_allclose(vowel_peaks('a'),
                               np.array([730, 1090, 2440]),
                               atol=bin_width)
    np.testing.assert_allclose(vowel_peaks('i'),
                               np.array([270, 2290, 3010]),
                               atol=bin_width)
    np.testing.assert_allclose(vowel_peaks('u'),
                               np.array([300, 870, 2240]),
                               atol=bin_width)


if __name__ == '__main__':
  absltest.main()
