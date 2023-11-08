"""A python port of portions of the Matlab Auditory Toolbox.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from typing import List


def ErbSpace(low_freq: float = 100, high_freq: float = 44100/4,
             n: int = 100) -> np.ndarray:
  """This function computes an array of N frequencies uniformly spaced between
  high_freq and low_freq on an erb scale.  N is set to 100 if not specified.

  See also linspace, logspace, MakeErbCoeffs, MakeErbFilters.

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
             np.exp(np.arange(1, 1+n)*
                    (-np.log(high_freq + ear_q*min_bw) +
                     np.log(low_freq + ear_q*min_bw))/n) * (high_freq +
                                                             ear_q*min_bw))
  return cf_array


def MakeErbFilters(fs: float, num_channels: int,
                   low_freq: float = 20) -> List[np.ndarray]:
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
  input argument is a vector, then the values of this vector are taken to
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
  pi = np.pi
  abs = np.abs  # pylint: disable=redefined-builtin
  sqrt = np.sqrt
  sin = np.sin
  cos = np.cos
  exp = np.exp
  i = np.array([1j], dtype=np.csingle)

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

  allfilts = np.ones(len(cf))
  fcoefs = [a0*allfilts, a11, a12, a13, a14, a2*allfilts,
            b0*allfilts, b1, b2, gain]
  return fcoefs


def ErbFilterBank(x: np.ndarray, fcoefs: List[np.ndarray]) -> np.ndarray:
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

  sos = np.stack((np.stack([a0/gain,  a0,   a0, a0], axis=1),
                  np.stack([a11/gain, a12, a13, a14], axis=1),
                  np.stack([a2/gain,  a2,   a2, a2], axis=1),
                  np.stack([b0, b0, b0, b0], axis=1),
                  np.stack([b1, b1, b1, b1], axis=1),
                  np.stack([b2, b2, b2, b2], axis=1)),
                axis=2)

  all_y = None
  for c in range(n_chan):
    y = signal.sosfilt(sos[c, :, :], x)
    if all_y is None:
      all_y = np.zeros((n_chan, len(y)), dtype=x.dtype)
    all_y[c, :] = y
  return all_y


def CorrelogramFrame(data: np.ndarray, pic_width: int,
                     start: int = 0, win_len: int = 0) -> np.ndarray:
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
  channels, data_len = data.shape
  if not win_len:
    win_len = data_len

  # Round up to double the window size, and then the next power of 2.
  fft_size = int(2**np.ceil(np.log2(2*max(pic_width, win_len))))

  start = max(0, start)
  last = min(data_len, start+win_len)
  a = .54
  b = -.46
  wr = math.sqrt(64/256)
  phi = np.pi/win_len
  ws = 2*wr/np.sqrt(4*a*a+2*b*b)*(
    a + b*np.cos(2*np.pi*(np.arange(win_len))/win_len + phi))

  f = np.zeros((channels, fft_size), dtype=data.dtype)
  f[:, :last-start] = data[:, start:last] * ws[:last-start]
  f = np.fft.fft(f, axis=1)
  f = np.fft.ifft(f*np.conj(f), axis=1)
  pic = np.maximum(0, np.real(f[:, :pic_width]))
  good_rows = np.logical_and( # Make sure first column is bigger than the rest.
      pic[:, 0] > 0,
      np.logical_and(pic[:, 0] > pic[:, 1], pic[:, 0] > pic[:, 2]))
  pic = np.where(np.expand_dims(good_rows, axis=-1),
                  pic / np.tile(np.sqrt(pic[:, :1]), (1, pic_width)),
                  np.array([0]))

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
  points = np.arange(kmax)
  points = (fs/freq)*(points-(
    fm_amp/(2*np.pi*fm_freq))*np.sin(2*np.pi*(fm_freq/freq)*points))
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
  y = np.zeros(sample_len, float)
  if isinstance(pitch, (int, float)):
    points = np.arange(0, sample_len, sample_rate/pitch)
  else:
    points = np.sort(np.asarray(pitch))
    points = points[points < sample_len-1]
  indices = np.floor(points).astype(int)

  #  Use a triangular approximation to an impulse function.  The important
  #  part is to keep the total amplitude the same.
  y[indices] = (indices+1)-points
  y[indices+1] = points-indices

  #  GlottalFilter(x,fs) - Filter an impulse train and simulate the glottal
  #    transfer function.  The sampling interval (sample_rate) is given in Hz.
  #    The filtering performed by this function is two first-order filters
  #    at 250Hz.
  a = np.exp(-250*2*np.pi/sample_rate)
  #y=filter([1,0,-1],[1,-2*a,a*a],y)      #  Not as good as one below....
  y = signal.lfilter([1],[1,0,-a*a],y)

  #  FormantFilter(input, f, fs) - Filter an input sequence to model one
  #    formant in a speech signal.  The formant frequency (in Hz) is given
  #    by f and the bandwidth of the formant is a constant 50Hz.  The
  #    sampling frequency in Hz is given by fs.
  if f1 > 0:
    cft = f1/sample_rate
    bw = 50
    q = f1/bw
    rho = np.exp(-np.pi * cft / q)
    theta = 2 * np.pi * cft * np.sqrt(1-1/(4 * q*q))
    a2 = -2*rho*np.cos(theta)
    a3 = rho*rho
    y=signal.lfilter([1+a2+a3],[1,a2,a3],y)

  #  FormantFilter(input, f, fs) - Filter an input sequence to model one
  #    formant in a speech signal.  The formant frequency (in Hz) is given
  #    by f and the bandwidth of the formant is a constant 50Hz.  The
  #    sampling frequency in Hz is given by fs.
  if f2 > 0:
    cft = f2/sample_rate
    bw = 50
    q = f2/bw
    rho = np.exp(-np.pi * cft / q)
    theta = 2 * np.pi * cft * np.sqrt(1-1/(4 * q*q))
    a2 = -2*rho*np.cos(theta)
    a3 = rho*rho
    y= signal.lfilter([1+a2+a3],[1,a2,a3],y)

  #  FormantFilter(input, f, fs) - Filter an input sequence to model one
  #    formant in a speech signal.  The formant frequency (in Hz) is given
  #    by f and the bandwidth of the formant is a constant 50Hz.  The
  #    sampling frequency in Hz is given by fs.
  if f3 > 0:
    cft = f3/sample_rate
    bw = 50
    q = f3/bw
    rho = np.exp(-np.pi * cft / q)
    theta = 2 * np.pi * cft * np.sqrt(1-1/(4 * q*q))
    a2 = -2*rho*np.cos(theta)
    a3 = rho*rho
    y= signal.lfilter([1+a2+a3],[1,a2,a3],y)
  return y


def CorrelogramArray(data, sr=16000, frame_rate=12, width=256):
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
  movie = None
  for i in range(frame_count):
    start = i*frame_increment
    frame = CorrelogramFrame(data, width, start, frame_increment*4)
    if movie is None:
      movie = np.zeros((frame_count, frame.shape[0],
                        frame.shape[1]), dtype=float)
    movie[i, :, :] = frame
  return movie

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

  drop_low = int(sr/high_pitch)
  if low_pitch > 0:
    drop_high = int(min(width,math.ceil(sr/low_pitch)))
  else:
    drop_high = width

  frames = correlogram.shape[0]

  pitch = np.zeros(frames)
  salience = np.zeros(frames)
  for j in range(frames):
    # Get one frame from the correlogram and compute
    # the sum (as a function of time lag) across all channels.
    summary = np.sum(correlogram[j, :, :], axis=0)
    zero_lag = summary[0]
    # Now we need to find the first pitch past the peak at zero
    # lag.  The following lines smooth the summary pitch a bit, then
    # look for the first point where the summary goes back up.
    # Everything up to this point is zeroed out.
    window_length = 16
    sumfilt = signal.lfilter(np.ones(window_length), [1,] , summary)
    sumdif = sumfilt[1:width] - sumfilt[:width-1]
    sumdif[:window_length] = 0
    valleys = np.argwhere(sumdif>0)
    summary[:int(valleys[0, 0])] = 0
    summary[1:drop_low] = 0
    summary[drop_high:] = 0
    # plt.plot(summary)
    # Now find the location of the biggest peak and call this the pitch
    p = np.argmax(summary)
    if p > 0:
      pitch[j] = sr/float(p)
    salience[j] = summary[p]/zero_lag

  return pitch,salience

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

  freqs = np.zeros(total_filters+2)
  freqs[:linear_filters] = (lowest_frequency +
                           np.arange(linear_filters)*linear_spacing)
  freqs[linear_filters:total_filters+2] = (
    freqs[linear_filters-1] * log_spacing**np.arange(1, log_filters+3))
  # print('freqs:', freqs)
  lower = freqs[:total_filters]
  center = freqs[1:total_filters+1]
  upper = freqs[2:total_filters+2]

  # We now want to combine FFT bins so that each filter has unit
  # weight, assuming a triangular weighting function.  First figure
  # out the height of the triangle, then we can figure out each
  # frequencies contribution
  mfcc_filter_weights = np.zeros((total_filters,fft_size))
  triangle_height = 2/(upper-lower)
  fft_freqs = np.arange(fft_size)/fft_size*sampling_rate

  for chan in range(total_filters):
    mfcc_filter_weights[chan,:] = (
        np.logical_and(fft_freqs > lower[chan], fft_freqs <= center[chan]) *
        triangle_height[chan]*(fft_freqs-lower[chan])/(center[chan]-
                                                       lower[chan]) +
        np.logical_and(fft_freqs > center[chan], fft_freqs < upper[chan]) *
        triangle_height[chan]*(upper[chan]-fft_freqs)/(upper[chan]-
                                                       center[chan]))

  if debug:
    plt.semilogx(fft_freqs,mfcc_filter_weights.T)
    #axis([lower(1) upper(total_filters) 0 max(max(mfcc_filter_weights))])

  ham_window = 0.54 - 0.46*np.cos(2*np.pi*np.arange(window_size)/window_size)

  if False:					# Window it like ComplexSpectrum  # pylint: disable=using-constant-test
    window_step = sampling_rate/frame_rate
    a = .54
    b = -.46
    wr = np.sqrt(window_step/window_size)
    phi = np.pi/window_size
    ham_window = (2*wr/np.sqrt(4*a*a+2*b*b)*
                 (a + b*np.cos(2*np.pi*np.arange(window_size)/window_size +
                               phi)))

  # Figure out Discrete Cosine Transform.  We want a matrix
  # dct(i,j) which is total_filters x cepstral_coefficients in size.
  # The i,j component is given by
  #                cos( i * (j+0.5)/total_filters pi )
  # where we have assumed that i and j start at 0.

  cepstral_indices = np.reshape(np.arange(cepstral_coefficients),
                               (cepstral_coefficients, 1))
  filter_indices = np.reshape(2*np.arange(0, total_filters)+1,
                              (1, total_filters))
  cos_term = np.cos(np.matmul(cepstral_indices,
                             filter_indices)*np.pi/2/total_filters)

  mfcc_dct_matrix = 1/np.sqrt(total_filters/2)*cos_term
  mfcc_dct_matrix[0,:] = mfcc_dct_matrix[0,:] * np.sqrt(2)/2

  if debug:
    plt.imshow(mfcc_dct_matrix)
    plt.xlabel('Filter Coefficient')
    plt.ylabel('Cepstral Coefficient')

  # Filter the input with the preemphasis filter.  Also figure how
  # many columns of data we will end up with.
  if True:   # pylint: disable=using-constant-test
    pre_emphasized = signal.lfilter([1 -.97], 1, input_signal)
  else:
    pre_emphasized = input_signal

  window_step = sampling_rate/frame_rate
  cols = int((len(input_signal)-window_size)/window_step)

  # Allocate all the space we need for the output arrays.
  ceps = np.zeros((cepstral_coefficients, cols))
  freqresp = np.zeros((fft_size//2, cols))
  fb = np.zeros((total_filters, cols))

  # Invert the filter bank center frequencies.  For each FFT bin
  # we want to know the exact position in the filter bank to find
  # the original frequency response.  The next block of code finds the
  # integer and fractional sampling positions.
  if True:  # pylint: disable=using-constant-test
    fr = np.arange(fft_size//2)/(fft_size/2)*sampling_rate/2
    j = 0
    for i in np.arange(fft_size//2):
      if fr[i] > center[j+1]:
        j = j + 1
      j = min(j, total_filters-2)
      # if j > total_filters-2:
      #   j = total_filters-1
      fr[i] = min(total_filters-1-.0001,
                  max(0,j + (fr[i]-center[j])/(center[j+1]-center[j])))
    fri = fr.astype(int)
    frac = fr - fri

    freqrecon = np.zeros((fft_size//2, cols))
  fbrecon = np.zeros((total_filters, cols))
  # Ok, now let's do the processing.  For each chunk of data:
  #    * Window the data with a hamming window,
  #    * Shift it into FFT order,
  #    * Find the magnitude of the fft,
  #    * Convert the fft data into filter bank outputs,
  #    * Find the log base 10,
  #    * Find the cosine transform to reduce dimensionality.
  for start in np.arange(cols):
    first = round(start*window_step)        # Round added by Malcolm
    last = round(first + window_size)
    fft_data = np.zeros(fft_size)
    fft_data[:window_size] = pre_emphasized[first:last]*ham_window
    fft_mag = np.abs(np.fft.fft(fft_data))
    ear_mag = np.log10(np.matmul(mfcc_filter_weights, fft_mag.T))

    ceps[:,start] = np.matmul(mfcc_dct_matrix, ear_mag)
    freqresp[:,start] = fft_mag[:fft_size//2].T
    fb[:,start] = ear_mag
    fbrecon[:,start] = np.matmul(mfcc_dct_matrix[:cepstral_coefficients,:].T,
                                 ceps[:,start])

    if True:  # pylint: disable=using-constant-test
      f10 = 10**fbrecon[:,start]
      freqrecon[:,start] = sampling_rate/fft_size * (f10[fri]*(1-frac) +
                                                   f10[fri+1]*frac)

  # OK, just to check things, let's also reconstruct the original FB
  # output.  We do this by multiplying the cepstral data by the transpose
  # of the original DCT matrix.  This all works because we were careful to
  # scale the DCT matrix so it was orthonormal.
  fbrecon = np.matmul(mfcc_dct_matrix[:cepstral_coefficients,:].T, ceps)
  return ceps,freqresp,fb,fbrecon,freqrecon
