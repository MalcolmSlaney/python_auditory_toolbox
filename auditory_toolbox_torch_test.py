"""Code to test the auditory toolbox."""
from absl.testing import absltest
import torch
import torch.fft
import numpy as np
import auditory_toolbox_torch as pat


class AuditoryToolboxTests(absltest.TestCase):
    """Test cases for the filterbank."""

    def test_erb_space(self):
        """Test ERB space."""
        low_freq = 100.0
        high_freq = 44100/4.0
        num_channels = 100
        cf_array = pat.erb_space(low_freq=low_freq, high_freq=high_freq,
                                 n=num_channels)
        cf_array = cf_array.numpy().squeeze()
        self.assertLen(cf_array, num_channels)
        # Make sure low and high CF's are where we expect them to be.
        self.assertAlmostEqual(cf_array[-1], low_freq)
        self.assertLess(cf_array[0], high_freq)

    def test_make_erb_filters(self):
        """Test fcoefs"""
        # Ten channel ERB Filterbank.  Make sure return has the right size.
        # Will test coefficients when we test the filterbank.
        fs = 16000
        low_freq = 100
        num_chan = 10
        fcoefs = pat.make_erb_filters(fs, num_chan, low_freq)
        self.assertLen(fcoefs, 10)
        # Test all the filter coefficient array shapes
        a0, a11, a12, a13, a14, a2, b0, b1, b2, gain = fcoefs

        self.assertEqual(a0.numpy().shape, (num_chan, 1))
        self.assertEqual(a11.numpy().shape, (num_chan, 1))
        self.assertEqual(a12.numpy().shape, (num_chan, 1))
        self.assertEqual(a13.numpy().shape, (num_chan, 1))
        self.assertEqual(a14.numpy().shape, (num_chan, 1))
        self.assertEqual(a2.numpy().shape, (num_chan, 1))
        self.assertEqual(b0.numpy().shape, (num_chan, 1))
        self.assertEqual(b1.numpy().shape, (num_chan, 1))
        self.assertEqual(b2.numpy().shape, (num_chan, 1))
        self.assertEqual(gain.numpy().shape, (num_chan, 1))

    def test_erb_filterbank_peaks(self):
        """Test peaks."""
        impulse_len = 512
        x = torch.zeros(1, impulse_len, dtype=torch.float64)
        x[:, 0] = 1.0

        m = pat.ErbFilterBank(
            sampling_rate=16000, num_channels=10, lowest_frequency=100)
        y = m(x)
        y = y.detach().numpy()
        self.assertEqual(y.shape, (1, 10, impulse_len))
        self.assertAlmostEqual(np.max(y), 0.10657410, delta=0.01)

        resp = 20 * np.log10(np.abs(np.fft.fft(y, axis=-1)))
        # Test to make sure spectral peaks are in the right place for each
        # channel
        matlab_peak_locs = [184, 132, 94, 66, 46, 32, 21, 14, 8, 4]
        python_peak_locs = np.argmax(
            resp[..., : impulse_len // 2], axis=-1).squeeze()
        np.testing.assert_equal(matlab_peak_locs, python_peak_locs+1)

        self.assertEqual(resp.shape, torch.Size([1, 10, 512]))
        self.assertEqual(
            list(python_peak_locs.squeeze() + 1), matlab_peak_locs
        )

        matlab_temporal_peak_locs = [12, 13, 23, 32, 46, 51, 77, 122, 143, 164]
        python_temporal_peak_locs = np.argmax(y, axis=-1)
        self.assertEqual(
            list(python_temporal_peak_locs.squeeze() +
                 1), matlab_temporal_peak_locs
        )

    def test_fm_points(self):
        """Test fm points"""
        base_pitch = 160
        sample_rate = 16000
        fmfreq = 10
        fmamp = 20
        points = pat.fm_points(100000, base_pitch, fmfreq, fmamp, 16000)

        # Make sure the average glottal pulse locations is 1 over the pitch
        d_points = points[1:] - points[:-1]
        self.assertAlmostEqual(np.mean(d_points.numpy()),
                               sample_rate/base_pitch, delta=1)

    def test_make_vowel(self):
        """Test make vowels."""

        def local_peaks(x):
            i = np.argwhere(np.logical_and(x[:-2] < x[1:-1],
                                           x[2:] < x[1:-1])) + 1
            return [j[0] for j in i]

        test_seq = local_peaks(np.array([1, 2, 3, 2, 1, 1, 2, 2, 3, 4, 1]))
        np.testing.assert_equal(test_seq, np.array([2, 9]))

        def vowel_peaks(vowel):
            """Find the frequencies of the spectral peaks."""
            sample_rate = 16000
            vowel = pat.make_vowel(1024, [1,], sample_rate, vowel)
            vowel = vowel.numpy()
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

    def test_erb_filterbank_output_shapes(self):
        """Test output shapes."""
        x1 = torch.zeros(64, 512, dtype=torch.float64)
        x1[:, 0] = 1.0
        x2 = torch.zeros(1, 512, dtype=torch.float64)
        x2[0, 0] = 1.0

        m = pat.ErbFilterBank(
            sampling_rate=16000, num_channels=10, lowest_frequency=100)

        y1 = m(x1)
        y2 = m(x2)

        assert np.isclose(y1.detach().numpy(), y2.detach().numpy()).all()
        self.assertEqual(list(y1.shape), [64, 10, 512])
        self.assertEqual(list(y2.shape), [1, 10, 512])
        self.assertAlmostEqual(
            torch.sum(torch.abs((y1 - y2))), torch.tensor(0.0)
        )

        y = m(torch.zeros(5, 2, 3, 10000, dtype=torch.float64))
        self.assertEqual(list(y.shape), [5, 2, 3, 10, 10000])

    def test_erb_filterbank_dtype(self):
        """Test data type."""
        x = torch.rand(5, 2, 3, 1000, dtype=torch.float32)
        m = pat.ErbFilterBank(
            sampling_rate=44100, num_channels=10, lowest_frequency=100
        )
        m.to(dtype=torch.float32)
        y = m(x)
        self.assertEqual(list(y.shape), [5, 2, 3, 10, 1000])

    def test_erb_filterbank_num_channels(self):
        """Test shapes with different number of channels."""
        x = torch.randn(64, 512, dtype=torch.float64)

        m1 = pat.ErbFilterBank(
            sampling_rate=16000, num_channels=10, lowest_frequency=100
        )
        m2 = pat.ErbFilterBank(
            sampling_rate=16000, num_channels=32, lowest_frequency=100
        )
        m3 = pat.ErbFilterBank(
            sampling_rate=16000, num_channels=64, lowest_frequency=100
        )

        self.assertEqual(list(m1(x).shape), [64, 10, 512])
        self.assertEqual(list(m2(x).shape), [64, 32, 512])
        self.assertEqual(list(m3(x).shape), [64, 64, 512])

    def test_make_vowels_peaks_i(self):
        """Test peaks /i/"""
        wav_len = 8000
        loc = np.zeros(4)
        for p, pitch in enumerate([50., 100., 512., 1024.]):
            y = pat.make_vowel(wav_len, pitch, sample_rate=16000,
                               f='i').numpy()
            y = y - np.mean(y)
            y_fft = np.fft.fft(y)
            loc[p] = np.argmax(20*np.log10(np.abs(y_fft[:wav_len//2])))

        self.assertEqual(list(loc.astype(np.int16)+1), [126, 151, 257, 1537])

    def test_make_vowels_peaks_u(self):
        """Test peaks /u/"""
        wav_len = 8000
        loc = np.zeros(4)
        for p, pitch in enumerate([50., 100., 512., 1024.]):
            y = pat.make_vowel(wav_len, pitch, sample_rate=16000,
                               f='u').numpy()
            y = y - np.mean(y)
            y_fft = np.fft.fft(y)
            loc[p] = np.argmax(20*np.log10(np.abs(y_fft[:wav_len//2])))

        self.assertEqual(list(loc.astype(np.int16)+1), [151, 151, 257, 513])

    def test_make_vowels_peaks_a(self):
        """Test peaks /a/"""
        wav_len = 8000
        loc = np.zeros(4)
        for p, pitch in enumerate([50., 100., 512., 1024.]):
            y = pat.make_vowel(wav_len, pitch, sample_rate=16000,
                               f='a').numpy()
            y = y - np.mean(y)
            y_fft = np.fft.fft(y)
            loc[p] = np.argmax(20*np.log10(np.abs(y_fft[:wav_len//2])))

        self.assertEqual(list(loc.astype(np.int16)+1), [376, 351, 513, 513])


if __name__ == "__main__":
    absltest.main()
