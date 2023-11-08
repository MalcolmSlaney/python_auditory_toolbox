"""A JAX port of portions of the Matlab Auditory Toolbox.
"""
import math

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from typing import List


def ErbSpace(low_freq: float = 100, high_freq: float = 44100/4,
             n: int = 100) -> jnp.ndarray:
  """This function computes an array of N frequencies uniformly spaced between
  high_freq and low_freq on an erb scale.  N is set to 100 if not specified.

  See also linspace, logspace, MakeerbCoeffs, MakeErbFilters.

  For a definition of erb, see Moore, B. C. J., and Glasberg, B. R. (1983).
  "Suggested formulae for calculating auditory-filter bandwidths and
  excitation patterns," J. Acoust. Soc. Am. 74, 750-753.

  Args:
    low_freq: The center frequency in Hz of the lowest channel
    high_freq: The upper limit in Hz of the channel bank.  The center frequency
      of the highest channel will be below this frequency.
    n: Number of channels

  Returns:
    An array of center frequencies, equally spaced on the ERB scale.
  """

  #  Change the following three parameters if you wish to use a different
  #  erb scale.  Must change in MakeerbCoeffs too.
  ear_q = 9.26449				# Glasberg and Moore Parameters
  min_bw = 24.7

  # All of the follow_freqing expressions are derived in Apple TR #35, "An
  # Efficient Implementation of the Patterson-Holdsworth Cochlear
  # Filter Bank."  See pages 33-34.
  cf_array = (-(ear_q*min_bw) +
             jnp.exp(jnp.arange(1, 1+n)*
                    (-jnp.log(high_freq + ear_q*min_bw) +
                     jnp.log(low_freq + ear_q*min_bw))/n) * (high_freq +
                                                             ear_q*min_bw))
  return cf_array


def MakeErbFilters(fs: float, num_channels: int,
                   low_freq:float = 20) -> List[jnp.ndarray]:
  """This function computes the filter coefficients for a bank of
  Gammatone filters.  These filters were defined by Patterson and
  Holdworth for simulating the cochlea.

  The result is returned as an array of filter coefficients.  Each row
  of the filter arrays contains the coefficients for four second order
  filters.  The transfer function for these four filters share the same
  denominator (poles) but have different numerators (zeros).  All of these
  coefficients are assembled into one vector that the ErbFilterBank
  can take apart to implement the filter.

  The filter bank contains "num_channels" channels that extend from
  half the sampling rate (fs) to "low_freq".  Alternatively, if the num_channels
  ijnp.t argument is a vector, then the values of this vector are taken to
  be the center frequency of each desired filter.  (The low_freq argument is
  ignored in this case.)

  Note this implementation fixes a problem in the original code by
  computing four separate second order filters.  This avoids a big
  problem with round off errors in cases of very small cfs (100Hz) and
  large sample rates (44kHz).  The problem is caused by roundoff error
  when a number of poles are combined, all very close to the unit
  circle.  Small errors in the eigth order coefficient, are multiplied
  when the eigth root is taken to give the pole location.  These small
  errors lead to poles outside the unit circle and instability.  Thanks
  to Julius Smith for leading me to the proper explanation.

  Execute the following code to evaluate the frequency
  response of a 10 channel filterbank.
    fcoefs = MakeErbFilters(16000,10,100)
    y = ErbFilterBank([1 zeros(1,511)], fcoefs)
   	resp = 20*log10(abs(fft(y')))
   	freqScale = (0:511)/512*16000
  	semilogx(freqScale(1:255),resp(1:255,:))
  	axis([100 16000 -60 0])
  	xlabel('Frequency (Hz)') ylabel('Filter Response (dB)')

  Args:
    fs: Sampling rate (in Hz) of the filterbank (needed to determine CFs).
    num_channel: How many channels in the filterbank.
    low_freq: The lowest center frequency of the filterbank.

  Returns: 
    A list of 11 num_channel-D arrays containing the filter parameters.
  """

  t = 1/fs
  if isinstance(num_channels, int):
    cf = ErbSpace(low_freq, fs/2, num_channels)
  else:
    cf = num_channels

  # So equations below match the original Matlab syntax
  pi = jnp.pi
  abs = jnp.abs  # pylint: disable=redefined-builtin
  sqrt = jnp.sqrt
  sin = jnp.sin
  cos = jnp.cos
  exp = jnp.exp
  i = jnp.array([1j], dtype=jnp.csingle)

  # Change the follow_freqing three parameters if you wish to use a different
  # erb scale.  Must change in ErbSpace too.
  ear_q = 9.26449				#  Glasberg and Moore Parameters
  min_bw = 24.7
  order = 1

  erb = ((cf/ear_q)**order + min_bw**order)**(1/order)
  b=1.019*2*pi*erb

  a0 = t
  a2 = 0
  b0 = 1
  b1 = -2*cos(2*cf*pi*t)/exp(b*t)
  b2 = exp(-2*b*t)

  a11 = -(2*t*cos(2*cf*pi*t)/exp(b*t) + 2*sqrt(3+2**1.5)*t*sin(2*cf*pi*t)/
      exp(b*t))/2
  a12 = -(2*t*cos(2*cf*pi*t)/exp(b*t) - 2*sqrt(3+2**1.5)*t*sin(2*cf*pi*t)/
      exp(b*t))/2
  a13 = -(2*t*cos(2*cf*pi*t)/exp(b*t) + 2*sqrt(3-2**1.5)*t*sin(2*cf*pi*t)/
      exp(b*t))/2
  a14 = -(2*t*cos(2*cf*pi*t)/exp(b*t) - 2*sqrt(3-2**1.5)*t*sin(2*cf*pi*t)/
      exp(b*t))/2

  gain = abs((-2*exp(4*i*cf*pi*t)*t +
                  2*exp(-(b*t) + 2*i*cf*pi*t)*t*
                          (cos(2*cf*pi*t) - sqrt(3 - 2**(3/2))*
                            sin(2*cf*pi*t))) *
            (-2*exp(4*i*cf*pi*t)*t +
              2*exp(-(b*t) + 2*i*cf*pi*t)*t*
                (cos(2*cf*pi*t) + sqrt(3 - 2**(3/2)) *
                sin(2*cf*pi*t)))*
            (-2*exp(4*i*cf*pi*t)*t +
              2*exp(-(b*t) + 2*i*cf*pi*t)*t*
                (cos(2*cf*pi*t) -
                sqrt(3 + 2**(3/2))*sin(2*cf*pi*t))) *
            (-2*exp(4*i*cf*pi*t)*t + 2*exp(-(b*t) + 2*i*cf*pi*t)*t*
            (cos(2*cf*pi*t) + sqrt(3 + 2**(3/2))*sin(2*cf*pi*t))) /
            (-2 / exp(2*b*t) - 2*exp(4*i*cf*pi*t) +
            2*(1 + exp(4*i*cf*pi*t))/exp(b*t))**4)

  allfilts = jnp.ones(len(cf))
  fcoefs = [a0*allfilts, a11, a12, a13, a14, a2*allfilts,
            b0*allfilts, b1, b2, gain]
  return fcoefs


def ErbFilterBank(x: jnp.ndarray, fcoefs: List[jnp.ndarray]) -> jnp.ndarray:
  """Filter an input signal with a filterbank, producing one output vector
  per channel.
  
  Args:
    x: The input signal, one-dimensional
    fcoefs: A list of 11 num-channel-dimensional arrays that describe the 
      filterbank.
  
  Returns:
    num-channel outputs in a num_channel x time array.
  """
  [a0, a11, a12, a13, a14, a2, b0, b1, b2, gain] = fcoefs
  n_chan = a0.shape[0]
  assert n_chan == a11.shape[0]
  assert n_chan == a12.shape[0]
  assert n_chan == a13.shape[0]
  assert n_chan == a14.shape[0]
  assert n_chan == b0.shape[0]
  assert n_chan == b1.shape[0]
  assert n_chan == gain.shape[0]

  sos = jnp.stack((jnp.stack([a0/gain,  a0,   a0, a0], axis=1),
                  jnp.stack([a11/gain, a12, a13, a14], axis=1),
                  jnp.stack([a2/gain,  a2,   a2, a2], axis=1),
                  jnp.stack([b0, b0, b0, b0], axis=1),
                  jnp.stack([b1, b1, b1, b1], axis=1),
                  jnp.stack([b2, b2, b2, b2], axis=1)),
                axis=2)

  def ErbKernel(f):
    return SosFilt(f, x)

  return jax.vmap(ErbKernel, in_axes=0)(sos)


def CorrelogramFrame(data: jnp.ndarray, pic_width: int,
                     start: int = 0, win_len: int = 0) -> jnp.ndarray:
  """Generate one from of a correlogram using FFTs to calculate autocorrelation.
  
  Args
    data: A num_channel x time array of input waveforms, one time domain signal
      per channel.
    pic_width: Number of pixels (time lags) in the final correlogram frame.
    start: The starting sample
    win_length: How much data to take from the input signal when computing the
      autocorrelation.

  Returns:
    A two dimensional array, of size num_channels x pic_width, containing one
    frame of the correlogram.
  """
  _, data_len = data.shape
  if not win_len:
    win_len = data_len

  # Round up to double the window size, and then the next power of 2.
  fft_size = int(2**jnp.ceil(jnp.log2(2*max(pic_width, win_len))))

  start = max(0, start)
  last = min(data_len, start+win_len)
  a = .54
  b = -.46
  wr = math.sqrt(64/256)
  phi = jnp.pi/win_len
  ws = 2*wr/jnp.sqrt(4*a*a+2*b*b)*(
    a + b*jnp.cos(2*jnp.pi*(jnp.arange(win_len))/win_len + phi))

  f = jnp.hstack((data[:, start:last] * ws[:last-start],
                  jnp.zeros((data.shape[0], fft_size - (last-start)))))
  f = jnp.fft.fft(f, axis=1)
  f = jnp.fft.ifft(f*jnp.conj(f), axis=1)
  pic = jnp.maximum(0, jnp.real(f[:, :pic_width]))
  good_rows = jnp.logical_and( # Make sure first column is bigger than the rest.
      pic[:, 0] > 0,
      jnp.logical_and(pic[:, 0] > pic[:, 1], pic[:, 0] > pic[:, 2]))
  pic = jnp.where(jnp.expand_dims(good_rows, axis=-1),
                  pic / jnp.tile(jnp.sqrt(pic[:, :1]), (1, pic_width)),
                  jnp.array([0]))

  return pic


def FMPoints(sample_len, freq, fm_freq=6, fm_amp=None, fs=22050):
  """Generate impulse train corresponding to a vibrato.

  points=FMPoints(sample_len, freq, fm_freq, fm_amp, fs)
  Generates (fractional) sample locations for frequency-modulated impulses
      sample_len         = number of samples
      freq        = pitch frequency (Hz)
      fm_freq      = vibrato frequency (Hz)  (defaults to 6 Hz)
      fm_amp       = max change in pitch  (defaults to 5% of freq)
      fs          = sample frequency     (defaults to 22254.545454 samples/s)

  Basic formula: phase angle = 2*pi*freq*t +
                          (fm_amp/fm_freq)*sin(2*pi*fm_freq*t)
      k-th zero crossing approximately at sample number
      (fs/freq)*(k - (fm_amp/(2*pi*fm_freq))*sin(2*pi*k*(fm_freq/freq)))

  Args: 
    sample_len: How much data to generate, in samples
    freq: Base frequency of the output signal (Hz)
    fm_freq: Vibrato frequency (in Hz)
    fm_amp: Magnitude of the FM deviation (in Hz)
    fs: Sample rate for the output signal.

  Returns:
    An impulse train, indicating the positive-going zero crossing 
    of the phase funcion.
  """
  if fm_amp is None:
    fm_amp = 0.05*freq

  kmax = int(freq*(sample_len/fs))
  points = jnp.arange(kmax)
  points = (fs/freq)*(points-(
    fm_amp/(2*jnp.pi*fm_freq))*jnp.sin(2*jnp.pi*(fm_freq/freq)*points))
  return points


def MakeVowel(sample_len, pitch, sample_rate, f1=0, f2=0, f3=0):
  """Synthesize an artificial vowel using formant filters.

   MakeVowel(sample_len, pitch [, sample_rate, f1, f2, f3]) -
   Make a vowel with
     "sample_len" samples and the given pitch.  The sample rate defaults to
     be 22254.545454 Hz (the native Mactinosh Sampling Rate).  The
     formant frequencies are f1, f2 & f3.  Some common vowels are
                Vowel       f1      f2      f3
                 /a/        730    1090    2440
                 /i/        270    2290    3010
                 /u/        300     870    2240

  The pitch variable can either be a scalar indicating the actual
       pitch frequency, or an array of impulse locations. Using an
       array of impulses allows this routine to compute vowels with
       varying pitch.

  Alternatively, f1 can be replaced with one of the following strings
       'a', 'i', 'u' and the appropriate formant frequencies are
       automatically selected.
  
  Args:
    sample_len: How many samples to generate
    pitch: Either a single floating point value indidcating a constant 
      pitch (in Hz), or a train of impulses generated by FMPoints.
    sample_rate: The sample rate for the output signal (Hz)
    f1: Either a vowel spec, one of /a/, /i/, or /u', or the frequency 
      of the first formatn.
    f2: Optional 2nd formant frequency (if f1 is not a vowel name)
    f3: Optional 3rd formant frequency (if f1 is not a vowel name)
    
  Returns:
    A time domain waveform containing the synthetic vowel sound.
  """
  if isinstance(f1, str):
    if f1 == 'a' or f1 == '/a/':
      f1, f2, f3 = (730, 1090, 2440)
    elif f1 == 'i' or f1 == '/i/':
      f1, f2, f3 = (270, 2290, 3010)
    elif f1 == 'u' or f1 == '/u/':
      f1, f2, f3 = (300, 870, 2240)


  #  GlottalPulses(pitch, fs, sample_len) - Generate a stream of
  #    glottal pulses with the given pitch (in Hz) and sampling
  #    frequency (sample_rate).  A vector of the requested length is returned.
  y = jnp.zeros(sample_len, float)
  if isinstance(pitch, (int, float)):
    points = jnp.arange(0, sample_len, sample_rate/pitch)
  else:
    points = jnp.sort(jnp.asarray(pitch))
    points = points[points < sample_len-1]
  indices = jnp.floor(points).astype(int)

  #  Use a triangular approximation to an impulse function.  The important
  #  part is to keep the total amplitude the same.
  y = y.at[indices].set((indices+1)-points)
  y = y.at[indices+1].set(points-indices)

  #  GlottalFilter(x,fs) - Filter an impulse train and simulate the glottal
  #    transfer function.  The sampling interval (sample_rate) is given in Hz.
  #    The filtering performed by this function is two first-order filters
  #    at 250Hz.
  a = jnp.exp(-250*2*jnp.pi/sample_rate)
  #y=filter([1,0,-1],[1,-2*a,a*a],y)      #  Not as good as one below....
  y = SignalFilter([1, 0, 0],[1,0,-a*a],y)

  #  FormantFilter(ijnp.t, f, fs) - Filter an ijnp.t sequence to model one
  #    formant in a speech signal.  The formant frequency (in Hz) is given
  #    by f and the bandwidth of the formant is a constant 50Hz.  The
  #    sampling frequency in Hz is given by fs.
  if f1 > 0:
    cft = f1/sample_rate
    bw = 50
    q = f1/bw
    rho = jnp.exp(-jnp.pi * cft / q)
    theta = 2 * jnp.pi * cft * jnp.sqrt(1-1/(4 * q*q))
    a2 = -2*rho*jnp.cos(theta)
    a3 = rho*rho
    y=SignalFilter([1+a2+a3, 0, 0],[1,a2,a3],y)

  #  FormantFilter(ijnp.t, f, fs) - Filter an ijnp.t sequence to model one
  #    formant in a speech signal.  The formant frequency (in Hz) is given
  #    by f and the bandwidth of the formant is a constant 50Hz.  The
  #    sampling frequency in Hz is given by fs.
  if f2 > 0:
    cft = f2/sample_rate
    bw = 50
    q = f2/bw
    rho = jnp.exp(-jnp.pi * cft / q)
    theta = 2 * jnp.pi * cft * jnp.sqrt(1-1/(4 * q*q))
    a2 = -2*rho*jnp.cos(theta)
    a3 = rho*rho
    y= SignalFilter([1+a2+a3, 0, 0],[1,a2,a3],y)

  #  FormantFilter(ijnp.t, f, fs) - Filter an ijnp.t sequence to model one
  #    formant in a speech signal.  The formant frequency (in Hz) is given
  #    by f and the bandwidth of the formant is a constant 50Hz.  The
  #    sampling frequency in Hz is given by fs.
  if f3 > 0:
    cft = f3/sample_rate
    bw = 50
    q = f3/bw
    rho = jnp.exp(-jnp.pi * cft / q)
    theta = 2 * jnp.pi * cft * jnp.sqrt(1-1/(4 * q*q))
    a2 = -2*rho*jnp.cos(theta)
    a3 = rho*rho
    y= SignalFilter([1+a2+a3, 0, 0],[1,a2,a3],y)
  return y


def CorrelogramArray(data: jnp.ndarray, sr: float = 16000,
                     frame_rate: int = 12, width: int = 256) -> jnp.ndarray:
  """Generate an array of correlogram frames.
  
  Args:
    data: The filterbank's output, size num_channel x time.
    sr: The sample rate for the data (needed when computing the frame times)
    frame_rate: How often (in Hz) correlogram frames should be generated.
    width: The width (in lags) of the correlogram

  Returns:
    A num_frames x num_channels x width tensor of correlogram frames.
  """
  _, sample_len = data.shape
  frame_increment = int(sr/frame_rate)
  frame_count = int((sample_len-width)/frame_increment) + 1
  movie = []
  for i in range(frame_count):
    start = i*frame_increment
    frame = CorrelogramFrame(data, width, start, frame_increment*4)
    movie.append(frame)
  return jnp.asarray(movie)

def CorrelogramPitch(correlogram, width, sr=22254.54,
                     low_pitch=0, high_pitch=20000):
  """Compute the summary of a correlogram to find the pitch.

  pitch=CorrelogramPitch(correlogram, width, sr, low_pitch, high_pitch
  computes the pitch of a correlogram sequence by finding the time lag
  with the largest correlation energy.

  Args:
    correlogram: A 3D correlogram array, output from CorrelogramArray.
      num_frames x num_channels x num_times
    width: Width of the correlogram.  Historical parameter. Should be
      equal to correlogram.shape[1]
    low_pitch: Lowest allowable pitch (Hz). Pitch peaks are only searched 
      within the region low_pitch to high_pitch
    high_pitch: Highest allowable pitch (Hz).

  Returns:
    A 2-element tuple, containing 
      1) a one-dimensional array of length num_frames indicating the pitch 
          or 0 if no pitch is found
      2) A one-dimensional array indicating the pitch salience on a scale
          from 0 (no pitch found) to 1 clear pitch.
  """
  assert correlogram.ndim == 3
  width = correlogram.shape[2] # Someday remove this unneeded parameter

  freqs = sr/jnp.arange(width)  # CF of each lag bin
  valid_pitch_lags = jnp.logical_and(freqs > low_pitch,
                                     freqs < high_pitch)

  pitch = []
  salience = []
  for j in range(correlogram.shape[0]):
    # Get one frame from the correlogram and compute
    # the sum (as a function of time lag) across all channels.
    summary = jnp.sum(correlogram[j, :, :], axis=0)
    zero_lag = summary[0]
    # Now we need to find the first pitch past the peak at zero
    # lag.  The following lines smooth the summary pitch a bit, then
    # look for the first point where the summary goes back up.
    # Everything up to this point is zeroed out.
    window_length = 16
    sumfilt = jnp.convolve(summary, jnp.ones(window_length)/window_length,
                           'same')

    # Find the local maximums in the filtered summary correlogram.
    local_peak = jnp.logical_and(sumfilt[1:-1] > sumfilt[0:-2],
                                 sumfilt[1:-1] > sumfilt[2:])
    local_peak = jnp.hstack((0, local_peak, 0))
    peaks = jnp.where(jnp.logical_and(local_peak,
                                      valid_pitch_lags),
                      summary,
                      0*summary)
    # Now find the location of the biggest peak and call this the pitch
    p = jnp.argmax(peaks)
    pitch.append(jnp.where(p > 0,
                           freqs[p],
                           0))
    salience.append(summary[p]/zero_lag)

  return jnp.array(pitch), jnp.array(salience)


def Mfcc(input_signal, sampling_rate=16000, frame_rate=100, debug=False):
  """Mfcc - Mel frequency cepstrum coefficient analysis.

  Find the cepstral coefficients (ceps) corresponding to the
  input.  
  
  Args:
    input_signal: The one-dimensional time-domain audio signal
    sampling_rate: The sample rate of the input in Hz.
    frame_rate: The desired output sampling rate
    debug: A debug flag that turns on various plots.

  Returns:
    A five-tuple consisting of:
    1) The MFCC representation, a 13 x num_frames output.
    2) The detailed fft magnitude (freqresp) used in MFCC calculation,
	  3) The mel-scale filter bank output (fb)
	  4) The filter bank output by inverting the cepstrals with a cosine
		    transform (fbrecon),
	  5) The smooth frequency response by interpolating the fb reconstruction
		  (freqrecon)

  Modified a bit to make testing an algorithm easier... 4/15/94
  Fixed Cosine Transform (indices of cos() were swapped) - 5/26/95
  Added optional frame_rate argument - 6/8/95
  Added proper filterbank reconstruction using inverse DCT - 10/27/95
  Added filterbank inversion to reconstruct spectrum - 11/1/95
  """
  #	Filter bank parameters
  lowest_frequency = 133.3333
  linear_filters = 13
  linear_spacing = 66.66666666
  log_filters = 27
  log_spacing = 1.0711703
  fft_size = 512
  cepstral_coefficients = 13
  window_size = 400
  window_size = 256		# Standard says 400, but 256 makes more sense
          # Really should be a function of the sample
          # rate (and the lowest_frequency) and the
          # frame rate.

  # Keep this around for later....
  total_filters = linear_filters + log_filters

  # Now figure the band edges.  Interesting frequencies are spaced
  # by linear_spacing for a while, then go logarithmic.  First figure
  # all the interesting frequencies.  Lower, center, and upper band
  # edges are all consequtive interesting frequencies.

  linear_freqs = (lowest_frequency +
                           jnp.arange(linear_filters)*linear_spacing)
  log_freqs = linear_freqs[-1] * log_spacing**jnp.arange(1,
                                                                log_filters+3)
  freqs = jnp.hstack((linear_freqs, log_freqs))
  lower = freqs[:total_filters]
  center = freqs[1:total_filters+1]
  upper = freqs[2:total_filters+2]

  # We now want to combine FFT bins so that each filter has unit
  # weight, assuming a triangular weighting function.  First figure
  # out the height of the triangle, then we can figure out each
  # frequencies contribution
  mfcc_filter_weights = jnp.zeros((total_filters,fft_size))
  triangle_height = 2/(upper-lower)
  fft_freqs = jnp.arange(fft_size)/fft_size*sampling_rate

  for chan in range(total_filters):
    mfcc_filter_weights = mfcc_filter_weights.at[chan,:].set(
        jnp.logical_and(fft_freqs > lower[chan], fft_freqs <= center[chan]) *
        triangle_height[chan]*(fft_freqs-lower[chan])/(center[chan]-
                                                       lower[chan]) +
        jnp.logical_and(fft_freqs > center[chan], fft_freqs < upper[chan]) *
        triangle_height[chan]*(upper[chan]-fft_freqs)/(upper[chan]-
                                                       center[chan]))

  if debug:
    plt.semilogx(fft_freqs,mfcc_filter_weights.T)
    #axis([lower(1) upper(total_filters) 0 max(max(mfcc_filter_weights))])

  ham_window = 0.54 - 0.46*jnp.cos(2*jnp.pi*jnp.arange(window_size)/window_size)

  if False:					# Window it like ComplexSpectrum  # pylint: disable=using-constant-test
    window_step = sampling_rate/frame_rate
    a = .54
    b = -.46
    wr = jnp.sqrt(window_step/window_size)
    phi = jnp.pi/window_size
    ham_window = (2*wr/jnp.sqrt(4*a*a+2*b*b)*
                 (a + b*jnp.cos(2*jnp.pi*jnp.arange(window_size)/window_size +
                               phi)))

  # Figure out Discrete Cosine Transform.  We want a matrix
  # dct(i,j) which is total_filters x cepstral_coefficients in size.
  # The i,j component is given by
  #                cos( i * (j+0.5)/total_filters pi )
  # where we have assumed that i and j start at 0.

  cepstral_indices = jnp.reshape(jnp.arange(cepstral_coefficients),
                               (cepstral_coefficients, 1))
  filter_indices = jnp.reshape(2*jnp.arange(0, total_filters)+1,
                              (1, total_filters))
  cos_term = jnp.cos(jnp.matmul(cepstral_indices,
                             filter_indices)*jnp.pi/2/total_filters)

  mfcc_dct_matrix = 1/jnp.sqrt(total_filters/2)*cos_term
  mfcc_dct_matrix = mfcc_dct_matrix.at[0,:].set(mfcc_dct_matrix[0,:] *
                                                jnp.sqrt(2)/2)

  if debug:
    plt.imshow(mfcc_dct_matrix)
    plt.xlabel('Filter Coefficient')
    plt.ylabel('Cepstral Coefficient')

  # Filter the ijnp.t with the preemphasis filter.  Also figure how
  # many columns of data we will end up with.
  if True:   # pylint: disable=using-constant-test
    pre_emphasized = SignalFilter([1, -.97, 0], [1, 0, 0], input_signal)
  else:
    pre_emphasized = input_signal

  window_step = sampling_rate/frame_rate
  cols = int((len(input_signal)-window_size)/window_step)

  # Allocate all the space we need for the output arrays.
  ceps = []
  freqresp = []
  fb = []

  # Invert the filter bank center frequencies.  For each FFT bin
  # we want to know the exact position in the filter bank to find
  # the original frequency response.  The next block of code finds the
  # integer and fractional sampling positions.
  if True:  # pylint: disable=using-constant-test
    fr = jnp.arange(fft_size//2)/(fft_size/2)*sampling_rate/2
    j = 0
    for i in jnp.arange(fft_size//2):
      if fr[i] > center[j+1]:
        j = j + 1
      j = min(j, total_filters-2)
      # if j > total_filters-2:
      #   j = total_filters-1
      fr = fr.at[i].set(min(total_filters-1-.0001,
                            max(0,j + (fr[i]-center[j])/(center[j+1]-
                                                         center[j]))))
    fri = fr.astype(int)
    frac = fr - fri

  freqrecon = []
  fbrecon = []
  # Ok, now let's do the processing.  For each chunk of data:
  #    * Window the data with a hamming window,
  #    * Shift it into FFT order,
  #    * Find the magnitude of the fft,
  #    * Convert the fft data into filter bank outputs,
  #    * Find the log base 10,
  #    * Find the cosine transform to reduce dimensionality.
  for start in jnp.arange(cols):
    first = round(start*window_step)        # Round added by Malcolm
    last = round(first + window_size)
    fft_data = jnp.zeros(fft_size)
    fft_data = fft_data.at[:window_size].set(pre_emphasized[first:last]*
                                             ham_window)
    fft_mag = jnp.abs(jnp.fft.fft(fft_data))
    ear_mag = jnp.log10(jnp.matmul(mfcc_filter_weights, fft_mag.T))

    ceps.append(jnp.expand_dims(jnp.matmul(mfcc_dct_matrix, ear_mag), axis=-1))
    freqresp.append(jnp.expand_dims(fft_mag[:fft_size//2].T, axis=-1))
    fb.append(jnp.expand_dims(ear_mag, axis=-1))
    fbrecon.append(jnp.matmul(mfcc_dct_matrix[:cepstral_coefficients,:].T,
                              ceps[-1]))

    f10 = 10**fbrecon[-1]
    recon = sampling_rate/fft_size * (f10[fri, 0]*(1-frac) +
                                      f10[fri+1, 0]*frac)
    freqrecon.append(jnp.expand_dims(recon, axis=-1))
  ceps = jnp.hstack(ceps)
  freqresp = jnp.hstack(freqresp)
  fb = jnp.hstack(fb)
  fbrecon = jnp.hstack(fbrecon)
  freqrecon = jnp.hstack(freqrecon)

  # OK, just to check things, let's also reconstruct the original FB
  # output.  We do this by multiplying the cepstral data by the transpose
  # of the original DCT matrix.  This all works because we were careful to
  # scale the DCT matrix so it was orthonormal.
  fbrecon = jnp.matmul(mfcc_dct_matrix[:cepstral_coefficients,:].T, ceps)
  return ceps, freqresp, fb, fbrecon, freqrecon


@jax.jit
def FilterScan(carry, x, a, b):
  """Internal function needed for jax.lax.scan."""
  vzm1, vzm2 = carry
  v =        x - a[1]*vzm1 - a[2]*vzm2
  y = b[0] * v + b[1]*vzm1 + b[2]*vzm2
  vzm2 = vzm1   # v delayed by one sample
  vzm1 = v      # v delayed by two samples
  carry = vzm1, vzm2
  return carry, y


@jax.jit
def SignalFilter(b, a, x):
  """Redefine the filter function in scipy.signal.lfiter.  This version only
  does second-order sections, and always filters over the last dimension."""
  b = jnp.asarray(b) / a[0]
  a = jnp.asarray(a) / a[0]

  # Define a scan function with the filter parameters.
  def FilterKernel(carry, x):
    return FilterScan(carry, x, a, b)

  _, y = jax.lax.scan(FilterKernel, (0.0, 0.0), x)
  return y


@jax.jit
def SosFilt(sos, x):
  """Redefine the sosfilt function from scipy.signal.sosfilter.  This version
  only filters over the last dimension.
  """
  stages, six = sos.shape
  assert six == 6

  for i in range(stages):
    x = SignalFilter(sos[i, :3], sos[i, 3:], x)
  return x
