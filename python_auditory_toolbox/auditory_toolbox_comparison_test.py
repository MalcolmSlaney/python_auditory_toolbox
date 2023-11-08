""" Direct comparison of the three implementations."""
from absl.testing import absltest
import torch
import torch.fft
import numpy as np
import jax.numpy as jnp

import auditory_toolbox_torch as pat_torch
import auditory_toolbox as pat_np
import auditory_toolbox_jax as pat_jax

class AuditoryToolboxTests(absltest.TestCase):

  """Compare the three different auditory toolbox implementations."""

  def test_erbfilter_fcoefs(self):
    def compare_fcoefs(num_chan,fs,low_freq):
      fcoefs_np = pat_np.MakeErbFilters(fs, num_chan, low_freq)
      fcoefs_jax = pat_jax.MakeErbFilters(fs, num_chan, low_freq)
      model_torch = pat_torch.ErbFilterBank(fs, num_chan, low_freq)

      # Compare fcoefs
      fcoefs_np = np.c_[fcoefs_np].T
      fcoefs_jax = np.asarray(jnp.c_[fcoefs_jax]).T
      fcoefs_torch = torch.cat(model_torch.fcoefs,dim=1).numpy()

      np.testing.assert_almost_equal(fcoefs_np, fcoefs_torch)
      np.testing.assert_almost_equal(fcoefs_jax, fcoefs_np, decimal=6)
      np.testing.assert_almost_equal(fcoefs_jax, fcoefs_torch, decimal=6)

    compare_fcoefs(10, 16000, 60)
    compare_fcoefs(64, 16000, 60)
    compare_fcoefs(128, 16000, 60)
    compare_fcoefs(128, 44100, 60)

  def test_erbfilter_output(self):
    def compare_erbfilterbank_output(input_vec, num_chan, fs, low_freq):
      x = input_vec
      fcoefs_np = pat_np.MakeErbFilters(fs, num_chan, low_freq)
      fcoefs_jax = pat_jax.MakeErbFilters(fs, num_chan, low_freq)
      model_torch = pat_torch.ErbFilterBank(fs, num_chan,low_freq)

      y_np = pat_np.ErbFilterBank(x, fcoefs_np)
      y_jax = np.asarray(pat_np.ErbFilterBank(x, fcoefs_jax))
      y_torch = model_torch(torch.tensor(x).unsqueeze(0)).numpy().squeeze()

      # Compare output
      np.testing.assert_almost_equal(y_np, y_torch)
      np.testing.assert_almost_equal(y_np, y_jax, decimal=4)
      np.testing.assert_almost_equal(y_torch, y_jax, decimal=4)

    # Simulation 1
    x = np.zeros(512)
    x[0] = 1
    compare_erbfilterbank_output(x, 10, 16000, 100)
    compare_erbfilterbank_output(x, 64, 16000, 100)
    compare_erbfilterbank_output(x, 128, 44100, 100)

    # Simulation 2
    compare_erbfilterbank_output(np.random.randn(10000), 10, 16000, 100)

    # Simulation 2
    sample_len = 20000
    sample_rate = 22254
    pitch_center = 120
    x = pat_np.MakeVowel(sample_len, pat_np.FMPoints(sample_len, pitch_center),
                      sample_rate, 'u')
    compare_erbfilterbank_output(x, 10, sample_rate, 100)
    compare_erbfilterbank_output(x, 64, sample_rate, 100)
    compare_erbfilterbank_output(x, 128, sample_rate, 100)

  def test_fmpoints(self):
    def compare_fm_points_outputs(sample_len, freq, fm_freq, fm_amp, fs):
      points_np = pat_np.FMPoints(sample_len, freq,
                                  fm_freq, fm_amp, fs)
      points_jax = pat_jax.FMPoints(sample_len, freq,
                                    fm_freq, fm_amp, fs)
      points_torch = pat_torch.fm_points(sample_len, freq,
                                         fm_freq, fm_amp, fs)
      points_jax = np.asarray(points_jax)
      points_torch = points_torch.squeeze().numpy()
      np.testing.assert_almost_equal(points_np, points_torch)
      np.testing.assert_almost_equal(points_np, points_jax, decimal=3)
      np.testing.assert_almost_equal(points_jax, points_torch, decimal=3)

    base_pitch = 160
    sample_rate = 16000
    fmfreq = 10
    fmamp = 20
    sample_len = 10000
    compare_fm_points_outputs(sample_len,base_pitch,
                              fmfreq,fmamp,sample_rate)

    base_pitch = 160
    sample_rate = 16000
    fmfreq = 100
    fmamp = 20
    sample_len = 10000
    compare_fm_points_outputs(sample_len,base_pitch,
                              fmfreq,fmamp,sample_rate)

    base_pitch = 560
    sample_rate = 16000
    fmfreq = 50
    fmamp = 20
    sample_len = 10000
    compare_fm_points_outputs(sample_len,base_pitch,
                              fmfreq,fmamp,sample_rate)

  def test_correlogram_array(self):
    def compare_correlogram_output(input_vec, num_chan, fs, low_freq,
                                    frame_width):
      x = input_vec
      fcoefs_np = pat_np.MakeErbFilters(fs, num_chan, low_freq)
      fcoefs_jax = pat_jax.MakeErbFilters(fs, num_chan, low_freq)
      model_torch = pat_torch.ErbFilterBank(fs,num_chan,low_freq)

      y_np = pat_np.ErbFilterBank(x, fcoefs_np)
      y_jax = pat_np.ErbFilterBank(x, fcoefs_jax)
      y_torch = model_torch(torch.tensor(x).unsqueeze(0))


      y_np = pat_np.CorrelogramFrame(y_np, frame_width)
      y_jax = pat_np.CorrelogramFrame(y_jax, frame_width)
      y_torch = pat_torch.correlogram_frame(y_torch, frame_width)
      y_torch = y_torch.squeeze().numpy()
      y_jax = np.asarray(y_jax)
      # Compare output
      np.testing.assert_almost_equal(y_np,y_torch)
      np.testing.assert_almost_equal(y_np,y_jax,decimal=3)
      np.testing.assert_almost_equal(y_torch,y_jax,decimal=3)

    # Simulation 1
    x = np.zeros(512)
    x[0] = 1
    compare_correlogram_output(x, 10, 16000, 100, 256)
    compare_correlogram_output(x, 64, 16000, 100, 256)
    compare_correlogram_output(x, 128, 44100, 100, 256)
    compare_correlogram_output(x, 10, 16000, 100, 512)
    compare_correlogram_output(x, 64, 16000, 100, 512)
    compare_correlogram_output(x, 128, 44100, 100, 512)

    # # Simulation 2
    sample_len = 20000
    sample_rate = 22254
    pitch_center = 120
    x = pat_np.MakeVowel(sample_len,
                         pat_np.FMPoints(sample_len, pitch_center),
                         sample_rate,
                         'u')
    compare_correlogram_output(x, 10, 16000, 100, 256)
    compare_correlogram_output(x, 64, 16000, 100, 256)
    compare_correlogram_output(x, 128, 44100, 100, 256)
    compare_correlogram_output(x, 10, 16000, 100, 512)
    compare_correlogram_output(x, 64, 16000, 100, 512)
    compare_correlogram_output(x, 128, 44100, 100, 512)


if __name__ == '__main__':
  absltest.main()
