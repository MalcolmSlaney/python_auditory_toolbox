import math

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from typing import List


def ERBSpace(lowFreq: float = 100, highFreq: float = 44100/4,
             n: int = 100) -> np.ndarray:
  """This function computes an array of N frequencies uniformly spaced between
  highFreq and lowFreq on an ERB scale.  N is set to 100 if not specified.

  See also linspace, logspace, MakeERBCoeffs, MakeERBFilters.

  For a definition of ERB, see Moore, B. C. J., and Glasberg, B. R. (1983).
  "Suggested formulae for calculating auditory-filter bandwidths and
  excitation patterns," J. Acoust. Soc. Am. 74, 750-753.
  """

  #  Change the following three parameters if you wish to use a different
  #  ERB scale.  Must change in MakeERBCoeffs too.
  EarQ = 9.26449;				# Glasberg and Moore Parameters
  minBW = 24.7;
  order = 1;

  # All of the followFreqing expressions are derived in Apple TR #35, "An
  # Efficient Implementation of the Patterson-Holdsworth Cochlear
  # Filter Bank."  See pages 33-34.
  cfArray = -(EarQ*minBW) + np.exp(np.arange(1, 1+n)*(-np.log(highFreq + EarQ*minBW) +
                                                 np.log(lowFreq + EarQ*minBW))/n) * (highFreq + EarQ*minBW)
  return cfArray


def MakeERBFilters(fs: float, numChannels: int, lowFreq: float) -> np.ndarray:
  """This function computes the filter coefficients for a bank of
  Gammatone filters.  These filters were defined by Patterson and
  Holdworth for simulating the cochlea.

  The result is returned as an array of filter coefficients.  Each row
  of the filter arrays contains the coefficients for four second order
  filters.  The transfer function for these four filters share the same
  denominator (poles) but have different numerators (zeros).  All of these
  coefficients are assembled into one vector that the ERBFilterBank
  can take apart to implement the filter.

  The filter bank contains "numChannels" channels that extend from
  half the sampling rate (fs) to "lowFreq".  Alternatively, if the numChannels
  input argument is a vector, then the values of this vector are taken to
  be the center frequency of each desired filter.  (The lowFreq argument is
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
    fcoefs = MakeERBFilters(16000,10,100)
    y = ERBFilterBank([1 zeros(1,511)], fcoefs)
   	resp = 20*log10(abs(fft(y')))
   	freqScale = (0:511)/512*16000;
  	semilogx(freqScale(1:255),resp(1:255,:))
  	axis([100 16000 -60 0])
  	xlabel('Frequency (Hz)') ylabel('Filter Response (dB)')

  Rewritten by Malcolm Slaney@Interval.  June 11, 1998.
  (c) 1998 Interval Research Corporation
  """

  T = 1/fs
  if isinstance(numChannels, int):
    cf = ERBSpace(lowFreq, fs/2, numChannels)
  else:
    cf = numChannels

  # So equations below match the original Matlab syntax
  pi = np.pi
  abs = np.abs
  sqrt = np.sqrt
  sin = np.sin
  cos = np.cos
  exp = np.exp
  i = np.array([1j], dtype=np.csingle)

  # Change the followFreqing three parameters if you wish to use a different
  # ERB scale.  Must change in ERBSpace too.
  EarQ = 9.26449;				#  Glasberg and Moore Parameters
  minBW = 24.7;
  order = 1;

  ERB = ((cf/EarQ)**order + minBW**order)**(1/order)
  B=1.019*2*pi*ERB;

  A0 = T;
  A2 = 0;
  B0 = 1;
  B1 = -2*cos(2*cf*pi*T)/exp(B*T)
  B2 = exp(-2*B*T)

  A11 = -(2*T*cos(2*cf*pi*T)/exp(B*T) + 2*sqrt(3+2**1.5)*T*sin(2*cf*pi*T)/
      exp(B*T))/2;
  A12 = -(2*T*cos(2*cf*pi*T)/exp(B*T) - 2*sqrt(3+2**1.5)*T*sin(2*cf*pi*T)/
      exp(B*T))/2;
  A13 = -(2*T*cos(2*cf*pi*T)/exp(B*T) + 2*sqrt(3-2**1.5)*T*sin(2*cf*pi*T)/
      exp(B*T))/2;
  A14 = -(2*T*cos(2*cf*pi*T)/exp(B*T) - 2*sqrt(3-2**1.5)*T*sin(2*cf*pi*T)/
      exp(B*T))/2;

  gain = abs((-2*exp(4*i*cf*pi*T)*T +
                  2*exp(-(B*T) + 2*i*cf*pi*T)*T*
                          (cos(2*cf*pi*T) - sqrt(3 - 2**(3/2))*
                            sin(2*cf*pi*T))) *
            (-2*exp(4*i*cf*pi*T)*T +
              2*exp(-(B*T) + 2*i*cf*pi*T)*T*
                (cos(2*cf*pi*T) + sqrt(3 - 2**(3/2)) *
                sin(2*cf*pi*T)))*
            (-2*exp(4*i*cf*pi*T)*T +
              2*exp(-(B*T) + 2*i*cf*pi*T)*T*
                (cos(2*cf*pi*T) -
                sqrt(3 + 2**(3/2))*sin(2*cf*pi*T))) *
            (-2*exp(4*i*cf*pi*T)*T + 2*exp(-(B*T) + 2*i*cf*pi*T)*T*
            (cos(2*cf*pi*T) + sqrt(3 + 2**(3/2))*sin(2*cf*pi*T))) /
            (-2 / exp(2*B*T) - 2*exp(4*i*cf*pi*T) +
            2*(1 + exp(4*i*cf*pi*T))/exp(B*T))**4)

  allfilts = np.ones(len(cf))
  fcoefs = [A0*allfilts, A11, A12, A13, A14, A2*allfilts, B0*allfilts, B1, B2, gain]
  return fcoefs


def ERBFilterBank(x: np.ndarray, fcoefs: List[np.ndarray]) -> np.ndarray:
  [A0, A11, A12, A13, A14, A2, B0, B1, B2, gain] = fcoefs
  x_len = len(x)
  n_chan = A0.shape[0]
  assert n_chan == A11.shape[0]
  assert n_chan == A12.shape[0]
  assert n_chan == A13.shape[0]
  assert n_chan == A14.shape[0]
  assert n_chan == B0.shape[0]
  assert n_chan == B1.shape[0]
  assert n_chan == gain.shape[0]

  sos = np.stack((np.stack([A0/gain,  A0,   A0, A0], axis=1),
                  np.stack([A11/gain, A12, A13, A14], axis=1),
                  np.stack([A2/gain,  A2,   A2, A2], axis=1),
                  np.stack([B0, B0, B0, B0], axis=1),
                  np.stack([B1, B1, B1, B1], axis=1),
                  np.stack([B2, B2, B2, B2], axis=1)),
                axis=2)

  all_y = None
  for c in range(n_chan):
    y = signal.sosfilt(sos[c, :, :], x)
    if all_y is None:
      all_y = np.zeros((n_chan, len(y)), dtype=x.dtype)
    all_y[c, :] = y
  return all_y


def CorrelogramFrame(data: np.ndarray, picWidth: int,
                     start: int = 0, winLen: int = 0) -> np.ndarray:
  channels, data_len = data.shape
  if not winLen:
    winLen = data_len

  pic = np.zeros((channels, start + winLen), dtype=data.dtype)
  fftSize = int(2**np.ceil(np.log2(max(picWidth, winLen))))

  start = max(0, start);
  last = min(data_len, start+winLen);
  a = .54;
  b = -.46;
  wr = math.sqrt(64/256);
  phi = np.pi/winLen;
  ws = 2*wr/np.sqrt(4*a*a+2*b*b)*(a + b*np.cos(2*np.pi*(np.arange(winLen))/winLen + phi));

  pic = np.zeros((channels, picWidth), dtype=data.dtype)
  for i in range(channels):
    f = np.zeros(fftSize);
    f[:last-start] = data[i, start:last] * ws[:last-start];
    f = np.fft.fft(f);
    f = np.fft.ifft(f*np.conj(f));
    pic[i, :] = np.real(f[:picWidth])
    if pic[i, 0] > 0 and pic[i, 0] > pic[i, 1] and pic[i,0] > pic[i, 2]:
      pic[i,:] = pic[i,:] / np.sqrt(pic[i,0])
    else:
      pic[i,:] = np.zeros(picWidth);
  return pic


def FMPoints(len, freq, fmFreq=6, fmAmp=None, fs=22050):
  """points=FMPoints(len, freq, fmFreq, fmAmp, fs)
  # Generates (fractional) sample locations for frequency-modulated impulses
  #     len         = number of samples
  #     freq        = pitch frequency (Hz)
  #     fmFreq      = vibrato frequency (Hz)  (defaults to 6 Hz)
  #     fmAmp       = max change in pitch  (defaults to 5% of freq)
  #     fs          = sample frequency     (defaults to 22254.545454 samples/s)
  #
  # Basic formula: phase angle = 2*pi*freq*t + (fmAmp/fmFreq)*sin(2*pi*fmFreq*t)
  #     k-th zero crossing approximately at sample number
  #     (fs/freq)*(k - (fmAmp/(2*pi*fmFreq))*sin(2*pi*k*(fmFreq/freq)))

  # (c) 1998 Interval Research Corporation
  """
  if fmAmp is None:
    fmAmp = 0.05*freq

  kmax = int(freq*(len/fs))
  points = np.arange(kmax)
  points = (fs/freq)*(points-(fmAmp/(2*np.pi*fmFreq))*np.sin(2*np/np.pi*(fmFreq/freq)*points))
  return points


def MakeVowel(len, pitch, sampleRate, f1=0, f2=0, f3=0):
  """
  #  MakeVowel(len, pitch [, sampleRate, f1, f2, f3]) - Make a vowel with
  #    "len" samples and the given pitch.  The sample rate defaults to
  #    be 22254.545454 Hz (the native Mactinosh Sampling Rate).  The
  #    formant frequencies are f1, f2 & f3.  Some common vowels are
  #               Vowel       f1      f2      f3
  #                /a/        730    1090    2440
  #                /i/        270    2290    3010
  #                /u/        300     870    2240
  #
  # The pitch variable can either be a scalar indicating the actual
  #      pitch frequency, or an array of impulse locations. Using an
  #      array of impulses allows this routine to compute vowels with
  #      varying pitch.
  #
  # Alternatively, f1 can be replaced with one of the following strings
  #      'a', 'i', 'u' and the appropriate formant frequencies are
  #      automatically selected.
  #  Modified by R. Duda, 3/13/94

  # (c) 1998 Interval Research Corporation
  """
  if isinstance(f1, str):
    if f1 == 'a' or f1 == '/a/':
      f1=730; f2=1090; f3=2440;
    elif f1 == 'i' or f1 == '/i/':
      f1=270; f2=2290; f3=3010;
    elif f1 == 'u' or f1 == '/u/':
      f1=300; f2=870; f3=2240;


  #  GlottalPulses(pitch, fs, len) - Generate a stream of
  #    glottal pulses with the given pitch (in Hz) and sampling
  #    frequency (sampleRate).  A vector of the requested length is returned.
  y = np.zeros(len, float)
  if isinstance(pitch, (int, float)):
    points = np.arange(0, len, sampleRate/pitch)
  else:
    points = np.sorted(np.asarray(pitch))
    points = points[points < len-1]
  indices = np.floor(points).astype(int)

  #  Use a triangular approximation to an impulse function.  The important
  #  part is to keep the total amplitude the same.
  y[indices] = (indices+1)-points;
  y[indices+1] = points-indices;

  #  GlottalFilter(x,fs) - Filter an impulse train and simulate the glottal
  #    transfer function.  The sampling interval (sampleRate) is given in Hz.
  #    The filtering performed by this function is two first-order filters
  #    at 250Hz.
  a = np.exp(-250*2*np.pi/sampleRate);
  #y=filter([1,0,-1],[1,-2*a,a*a],y);      #  Not as good as one below....
  y = signal.lfilter([1],[1,0,-a*a],y);

  #  FormantFilter(input, f, fs) - Filter an input sequence to model one
  #    formant in a speech signal.  The formant frequency (in Hz) is given
  #    by f and the bandwidth of the formant is a constant 50Hz.  The
  #    sampling frequency in Hz is given by fs.
  if f1 > 0:
          cft = f1/sampleRate;
          bw = 50;
          q = f1/bw;
          rho = np.exp(-np.pi * cft / q);
          theta = 2 * np.pi * cft * np.sqrt(1-1/(4 * q*q));
          a2 = -2*rho*np.cos(theta);
          a3 = rho*rho;
          y=signal.lfilter([1+a2+a3],[1,a2,a3],y);

  #  FormantFilter(input, f, fs) - Filter an input sequence to model one
  #    formant in a speech signal.  The formant frequency (in Hz) is given
  #    by f and the bandwidth of the formant is a constant 50Hz.  The
  #    sampling frequency in Hz is given by fs.
  if f2 > 0:
          cft = f2/sampleRate;
          bw = 50;
          q = f2/bw;
          rho = np.exp(-np.pi * cft / q);
          theta = 2 * np.pi * cft * np.sqrt(1-1/(4 * q*q));
          a2 = -2*rho*np.cos(theta);
          a3 = rho*rho;
          y= signal.lfilter([1+a2+a3],[1,a2,a3],y);

  #  FormantFilter(input, f, fs) - Filter an input sequence to model one
  #    formant in a speech signal.  The formant frequency (in Hz) is given
  #    by f and the bandwidth of the formant is a constant 50Hz.  The
  #    sampling frequency in Hz is given by fs.
  if f3 > 0:
          cft = f3/sampleRate;
          bw = 50;
          q = f3/bw;
          rho = np.exp(-np.pi * cft / q);
          theta = 2 * np.pi * cft * np.sqrt(1-1/(4 * q*q));
          a2 = -2*rho*np.cos(theta);
          a3 = rho*rho;
          y= signal.lfilter([1+a2+a3],[1,a2,a3],y)
  return y


def CorrelogramArray(data, sr=16000, frameRate=12, width=256):
  channels, len = data.shape
  frameIncrement = int(sr/frameRate)
  frameCount = int((len-width)/frameIncrement) + 1
  movie = None
  for i in range(frameCount):
    start = i*frameIncrement
    frame = CorrelogramFrame(data, width, start, frameIncrement*4)
    if movie is None:
      movie = np.zeros((frameCount, frame.shape[0], frame.shape[1]), dtype=float)
    movie[i, :, :] = frame
  return movie


#  mfcc - Mel frequency cepstrum coefficient analysis.
#   [ceps,freqresp,fb,fbrecon,freqrecon] = ...
#			mfcc(input, samplingRate, [frameRate])
# Find the cepstral coefficients (ceps) corresponding to the
# input.  Four other quantities are optionally returned that
# represent:
#	the detailed fft magnitude (freqresp) used in MFCC calculation, 
#	the mel-scale filter bank output (fb)
#	the filter bank output by inverting the cepstrals with a cosine 
#		transform (fbrecon),
#	the smooth frequency response by interpolating the fb reconstruction 
#		(freqrecon)
#  -- Malcolm Slaney, August 1993
# Modified a bit to make testing an algorithm easier... 4/15/94
# Fixed Cosine Transform (indices of cos() were swapped) - 5/26/95
# Added optional frameRate argument - 6/8/95
# Added proper filterbank reconstruction using inverse DCT - 10/27/95
# Added filterbank inversion to reconstruct spectrum - 11/1/95

# (c) 1998 Interval Research Corporation  

def mfcc(input, samplingRate=16000, frameRate=100, debug=False):
  # [r, c] = input.shape
  # if r > c:
  #   input=input.T
  
  #	Filter bank parameters
  lowestFrequency = 133.3333;
  linearFilters = 13;
  linearSpacing = 66.66666666;
  logFilters = 27;
  logSpacing = 1.0711703;
  fftSize = 512;
  cepstralCoefficients = 13;
  windowSize = 400;
  windowSize = 256;		# Standard says 400, but 256 makes more sense
          # Really should be a function of the sample
          # rate (and the lowestFrequency) and the
          # frame rate.

  # Keep this around for later....
  totalFilters = linearFilters + logFilters

  # Now figure the band edges.  Interesting frequencies are spaced
  # by linearSpacing for a while, then go logarithmic.  First figure
  # all the interesting frequencies.  Lower, center, and upper band
  # edges are all consequtive interesting frequencies. 

  freqs = np.zeros(totalFilters+2)
  freqs[:linearFilters] = lowestFrequency + np.arange(linearFilters)*linearSpacing;
  freqs[linearFilters:totalFilters+2] = freqs[linearFilters-1] * logSpacing**np.arange(1, logFilters+3);
  # print('freqs:', freqs)
  lower = freqs[:totalFilters];
  center = freqs[1:totalFilters+1];
  upper = freqs[2:totalFilters+2];

  # We now want to combine FFT bins so that each filter has unit
  # weight, assuming a triangular weighting function.  First figure
  # out the height of the triangle, then we can figure out each 
  # frequencies contribution
  mfccFilterWeights = np.zeros((totalFilters,fftSize))
  triangleHeight = 2/(upper-lower);
  fftFreqs = np.arange(fftSize)/fftSize*samplingRate;

  for chan in range(totalFilters):
    mfccFilterWeights[chan,:] = (
        np.logical_and(fftFreqs > lower[chan], fftFreqs <= center[chan]) * 
        triangleHeight[chan]*(fftFreqs-lower[chan])/(center[chan]-lower[chan]) + 
        np.logical_and(fftFreqs > center[chan], fftFreqs < upper[chan]) * 
        triangleHeight[chan]*(upper[chan]-fftFreqs)/(upper[chan]-center[chan]))
  
  if debug:
    plt.semilogx(fftFreqs,mfccFilterWeights.T)
    #axis([lower(1) upper(totalFilters) 0 max(max(mfccFilterWeights))])

  hamWindow = 0.54 - 0.46*np.cos(2*np.pi*np.arange(windowSize)/windowSize)

  if False:					# Window it like ComplexSpectrum
    windowStep = samplingRate/frameRate;
    a = .54;
    b = -.46;
    wr = np.sqrt(windowStep/windowSize);
    phi = np.pi/windowSize;
    hamWindow = 2*wr/np.sqrt(4*a*a+2*b*b)* (a + b*cos(2*np.pi*np.arange(windowSize)/windowSize + phi));

  # Figure out Discrete Cosine Transform.  We want a matrix
  # dct(i,j) which is totalFilters x cepstralCoefficients in size.
  # The i,j component is given by 
  #                cos( i * (j+0.5)/totalFilters pi )
  # where we have assumed that i and j start at 0.

  cepstralIndices = np.reshape(np.arange(cepstralCoefficients), (cepstralCoefficients, 1))
  filterIndices = np.reshape(2*np.arange(0, totalFilters)+1, (1, totalFilters))
  cosTerm = np.cos(np.matmul(cepstralIndices, filterIndices)*np.pi/2/totalFilters)
  
  mfccDCTMatrix = 1/np.sqrt(totalFilters/2)*cosTerm
  # mfccDCTMatrix = 1/np.sqrt(totalFilters/2)*np.cos(np.arange(cepstralCoefficients)) * 
  #         (2*(np.arange(totalFilters))+1) * np.pi/2/totalFilters)
  mfccDCTMatrix[0,:] = mfccDCTMatrix[0,:] * np.sqrt(2)/2;

  if debug:
    plt.imshow(mfccDCTMatrix)
    plt.xlabel('Filter Coefficient')
    plt.ylabel('Cepstral Coefficient')

  # Filter the input with the preemphasis filter.  Also figure how
  # many columns of data we will end up with.
  if True:
    preEmphasized = scipy.signal.lfilter([1 -.97], 1, input);
  else:
    preEmphasized = input;

  windowStep = samplingRate/frameRate;
  cols = int((len(input)-windowSize)/windowStep);

  # Allocate all the space we need for the output arrays.
  ceps = np.zeros((cepstralCoefficients, cols))
  freqresp = np.zeros((fftSize//2, cols))
  fb = np.zeros((totalFilters, cols))

  # Invert the filter bank center frequencies.  For each FFT bin
  # we want to know the exact position in the filter bank to find
  # the original frequency response.  The next block of code finds the
  # integer and fractional sampling positions.
  if True:
    fr = np.arange(fftSize//2)/(fftSize/2)*samplingRate/2;
    j = 0;
    for i in np.arange(fftSize//2):
      if fr[i] > center[j+1]:
        j = j + 1;
      j = min(j, totalFilters-2)
      # if j > totalFilters-2:
      #   j = totalFilters-1
      fr[i] = min(totalFilters-1-.0001, 
                  max(0,j + (fr[i]-center[j])/(center[j+1]-center[j])))
    fri = fr.astype(int)
    frac = fr - fri;

    freqrecon = np.zeros((fftSize//2, cols))
  fbrecon = np.zeros((totalFilters, cols))
  # Ok, now let's do the processing.  For each chunk of data:
  #    * Window the data with a hamming window,
  #    * Shift it into FFT order,
  #    * Find the magnitude of the fft,
  #    * Convert the fft data into filter bank outputs,
  #    * Find the log base 10,
  #    * Find the cosine transform to reduce dimensionality.
  for start in np.arange(cols):
      first = round(start*windowStep)        # Round added by Malcolm
      last = round(first + windowSize)
      fftData = np.zeros(fftSize)
      fftData[:windowSize] = preEmphasized[first:last]*hamWindow
      fftMag = np.abs(np.fft.fft(fftData));
      earMag = np.log10(np.matmul(mfccFilterWeights, fftMag.T))

      ceps[:,start] = np.matmul(mfccDCTMatrix, earMag)
      freqresp[:,start] = fftMag[:fftSize//2].T
      fb[:,start] = earMag
      fbrecon[:,start] = np.matmul(mfccDCTMatrix[:cepstralCoefficients,:].T, ceps[:,start])

      if True:
        f10 = 10**fbrecon[:,start]
        freqrecon[:,start] = samplingRate/fftSize * (f10[fri]*(1-frac) + f10[fri+1]*frac);

  # OK, just to check things, let's also reconstruct the original FB
  # output.  We do this by multiplying the cepstral data by the transpose
  # of the original DCT matrix.  This all works because we were careful to
  # scale the DCT matrix so it was orthonormal.
  if True:
    fbrecon = np.matmul(mfccDCTMatrix[:cepstralCoefficients,:].T, ceps)
  #	imagesc(mt(:,1:cepstralCoefficients)*mfccDCTMatrix);
  return ceps,freqresp,fb,fbrecon,freqrecon


