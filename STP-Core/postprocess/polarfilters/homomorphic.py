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
# Last modified: July, 8th 2016
#

from numpy import uint16, float32, iinfo, finfo, ndarray, log as nplog, exp as npexp, arange, meshgrid, sqrt
from numpy import copy, pad, zeros, median

#from numpy.fft import fft2, ifft2
from pyfftw import n_byte_align, simd_alignment
from pyfftw.interfaces.numpy_fft import rfft2, irfft2

def homomorphic(im, args):
	"""Process the input image with an homomorphic filtering.

	Parameters
	----------
	im : array_like
		Image data as numpy array.	

	d0 : float
		Cutoff in the range [0.01, 0.99] of the high pass Gaussian filter. 
        Higher values means more high pass effect. [Suggested for default: 0.80].

    alpha : float
		Offset to preserve the zero frequency where. Higher values means less 
        high pass effect. [Suggested for default: 0.2]

	(Parameters d0 and alpha have to passed as a string separated by ;)
		   
	Example (using tiffile.py)
	--------------------------
	>>> im = imread('im_orig.tif')
	>>> im = homomorphic(im, '0.5;0.2')    
	>>> imsave('im_flt.tif', im) 
	

	References
	----------
  

	"""    
	# Get args:
	param1, param2 = args.split(";")       	 
	d0 = (1.0 - float(param1))  # Simpler for user
	alpha = float(param2) 
	
	# Internal parameters for Gaussian low-pass filtering:
	d0 = d0 * (im.shape[1] / 2.0)	

	# Take the log:
	im = nplog(1 + im)

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

	# Prepare Guassian filter:
	H = npexp(-(D ** 2) / (2 * (d0 ** 2)))
	H = (1 - H) + alpha
	
	# Do the filtering:
	im = H * im   

	# Compute inverse FFT of the filtered data:
	n_byte_align(im, simd_alignment)
	#im = real(irfft2(im, threads=2))
	im = irfft2(im, threads=2)

	# Take the exp:
	im = npexp(im) - 1

	# Return image according to input type:
	return im.astype(float32)