###########################################################################
# (C) 2016 Elettra - Sincrotrone Trieste S.C.p.A.. All rights reserved.   #
#                                                                         #
#                                                                         #
# This file is part of STP-Core, the Python core of SYRMEP Tomo Project,  #
# a software tool for the reconstruction of experimental CT datasets.     #
#                                                                         #
# STP-Core is free software: you can redistribute it and/or modify it     #
# under the terms of the GNU General Public License as published by the   #
# Free Software Foundation, either version 3 of the License, or (at your  #
# option) any later version.                                              #
#                                                                         #
# STP-Core is distributed in the hope that it will be useful, but WITHOUT #
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or   #
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License    #
# for more details.                                                       #
#                                                                         #
# You should have received a copy of the GNU General Public License       #
# along with STP-Core. If not, see <http://www.gnu.org/licenses/>.        #
#                                                                         #
###########################################################################

#
# Author: Francesco Brun
# Last modified: August, 4th 2016
#

from numpy import float32, iinfo, finfo, ndarray, arange, meshgrid, sqrt
from numpy import exp, real, copy, zeros, ones, pad, ComplexWarning, hanning
from numpy import tile, concatenate
#from numpy.fft import fft2, ifft2
from pyfftw import n_byte_align, simd_alignment
from pyfftw.interfaces.numpy_fft import rfft2, irfft2

from warnings import simplefilter

def _windowing_lr(im, marg):
	
	vscale = ones(im.shape[1] - marg)

	hann = hanning(marg)
	vleft = hann[0:marg / 2]
	vright = hann[marg / 2:]
		
	vrow = concatenate((vleft,vscale), axis=1)
	vrow = concatenate((vrow,vright), axis=1)
	vmatrix = tile(vrow, (im.shape[0],1))

	# Correction for odd/even issues:
	marg = im.shape[1] - vmatrix.shape[1]
	tmp = zeros(vmatrix[:,vmatrix.shape[1] - 1].shape) # Get last column
	tmp = tile(tmp, (marg,1)) # Replicate the last column the right number of times
	vmatrix = concatenate((vmatrix,tmp.T), axis=1) # Concatenate tmp after the image

	# Apply smoothing:
	im = im * vmatrix

	return im.astype(float32)


def raven(im, args):
	"""Process a sinogram image with the Raven de-striping algorithm.
	
	Parameters
	----------
	im : array_like
		Image data as numpy array.

	n : int
		Size of the window (minimum n = 3) around the zero frequency where 
		filtering is actually applied (v0 parameter in Raven's article). Higher 
		values	means more smoothing effect. [Suggested for default: 3]

	d0 : float
		Cutoff in the range [0.01, 0.99] of the low pass filter (a Gaussian filter 
		is used instead of the originally proposed Butterworth filter in order to 
		have only one tuning parameter). Higher values means more smoothing effect. 
		[Suggested for default: 0.5].

	(Parameters n and d0 have to passed as a string separated by ;)
		   
	Example (using tiffile.py)
	--------------------------
	>>> im = imread('sino_orig.tif')
	>>> im = raven(im, '3;0.50')    
	>>> imsave('sino_flt.tif', im) 

	References
	----------
	C. Raven, Numerical removal of ring artifacts in microtomography,
	Review of Scientific Instruments 69(8):2978-2980, 1998.

	"""    
	# Disable a warning:
	simplefilter("ignore", ComplexWarning)
	
	# Get args:
	param1, param2 = args.split(";")    
	n = int(param1) 
	d0 = (1.0 - float(param2))  # Simpler for user
	
	# Internal parameters for Gaussian low-pass filtering:
	d0 = d0 * (im.shape[1] / 2.0)

	# Pad image:
	marg = im.shape
	im = pad(im, pad_width=((im.shape[0] / 2, im.shape[0] / 2), (0,0)), mode='reflect') # or 'constant' for zero padding
	im = pad(im, pad_width=((0,0) ,(im.shape[1] / 2, im.shape[1] / 2)), mode='edge')    # or 'constant' for zero padding

	# Windowing:
	im = _windowing_lr(im, marg[1])
	im = _windowing_lr(im.T, marg[0]).T	

	# Compute FFT:
	n_byte_align(im, simd_alignment) 
	im = rfft2(im, threads=2)

	# Prepare the frequency coordinates:
	u = arange(0, im.shape[0], dtype=float32)
	v = arange(0, im.shape[1], dtype=float32)

	# Compute the indices for meshgrid:
	u[(u > im.shape[0] / 2.0)] = u[(u > im.shape[0] / 2.0)] - im.shape[0]    
	v[(v > im.shape[1] / 2.0)] = v[(v > im.shape[1] / 2.0)] - im.shape[1]

	# Compute the meshgrid arrays:
	V, U = meshgrid(v, u)

	# Compute the distances D(U, V):
	D = sqrt(U ** 2 + V ** 2)

	# Prepare Guassian filter limited to a window around zero-frequency:
	H = exp(-(D ** 2) / (2 * (d0 ** 2)))
	if (n % 2 == 1):
		H[n / 2:-(n / 2),:] = 1
	else:
		H[n / 2:-(n / 2 - 1),:] = 1

	# Do the filtering:
	im = H * im   

	# Compute inverse FFT of the filtered data:
	n_byte_align(im, simd_alignment)
	#im = real(irfft2(im, threads=2))
	im = irfft2(im, threads=2)

	# Crop image:
	im = im[im.shape[0] / 4:(im.shape[0] / 4 + marg[0]), im.shape[1] / 4:(im.shape[1] / 4 + marg[1])]

	# Return image:
	return im.astype(float32)
