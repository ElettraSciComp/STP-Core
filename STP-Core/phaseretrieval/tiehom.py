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

from numpy import float32, complex64, arange, ComplexWarning, finfo
from numpy import copy, meshgrid, real
from numpy import pi, log as nplog, cos as npcos, sin as npsin

# (Un)comment the related lines to use either NumPY or PyFFTW:
#from numpy.fft import rfft2, irfft2
from pyfftw import n_byte_align, simd_alignment
#from pyfftw.interfaces.numpy_fft import fft2, ifft2
from pyfftw.interfaces.numpy_fft import rfft2, irfft2
from numpy.fft import fftshift, ifftshift
from warnings import simplefilter

from utils.padding import upperPowerOfTwo, padImage


def tiehom_plan(im, beta, delta, energy, distance, pixsize, padding):
	"""Pre-compute data to save time in further execution of phase_retrieval with TIE-HOM
	(Paganin's) algorithm.

	Parameters
	----------
	im : array_like
		Image data as numpy array. Only image size (shape) is actually used.

	beta : double
		Immaginary part of the complex X-ray refraction index.

	delta : double
		Decrement from unity of the complex X-ray refraction index.

	energy [KeV]: double
		Energy in KeV of the incident X-ray beam.

	distance [mm]: double
		Sample-to-detector distance in mm.

	pixsize [mm]: double
		Size in mm of the detector element.

	padding : bool
		Apply image padding to better process the boundary of the image
	
	"""
	# Get additional values:
	lam = (12.398424 * 10 ** (-7)) / energy # in mm
	mu = 4 * pi * beta / lam
		
	# Replicate pad image if required:
	dim0_o = im.shape[0]
	dim1_o = im.shape[1]
	if (padding):		
		n_pad0 = im.shape[0] + im.shape[0] / 2
		n_pad1 = im.shape[1] + im.shape[1] / 2
	else:
		n_pad0 = dim0_o
		n_pad1 = dim1_o

	# Ensure even size:
	if (n_pad0 % 2 == 1):
		n_pad0 = n_pad0 + 1
	if (n_pad1 % 2 == 1):
		n_pad1 = n_pad1 + 1

	# Set the transformed frequencies according to pixelsize:
	rows = n_pad0
	cols = n_pad1
	ulim = arange(-(cols) / 2, (cols) / 2)
	ulim = ulim * (2 * pi / (cols * pixsize))
	vlim = arange(-(rows) / 2, (rows) / 2)
	vlim = vlim * (2 * pi / (rows * pixsize))
	u,v = meshgrid(ulim, vlim)

	# Apply formula:
	den = 1 + distance * delta / mu * (u * u + v * v) + finfo(float32).eps # Avoids division by zero

	# Shift the denominator and get only real components (half of the
	# frequencies):
	den = fftshift(den)	
	den = den[:,0:den.shape[1] / 2 + 1] 
	
	return {'dim0':dim0_o, 'dim1':dim1_o ,'npad0':n_pad0, 'npad1':n_pad1, 'den':den , 'mu':mu }
	

def tiehom(im, plan, nr_threads=2):
	"""Process a tomographic projection image with the TIE-HOM (Paganin's) phase retrieval algorithm.

	Parameters
	----------
	im : array_like
		Flat corrected image data as numpy array.

	plan : structure
		Structure with pre-computed data (see tiehom_plan function).

	nr_threads : int 
		Number of threads to be used in the computation of FFT by PyFFTW (default = 2).
		
	"""
	# Extract plan values:
	dim0_o = plan['dim0']
	dim1_o = plan['dim1']
	n_pad0 = plan['npad0']
	n_pad1 = plan['npad1']
	marg0 = (n_pad0 - dim0_o) / 2
	marg1 = (n_pad1 - dim1_o) / 2
	den = plan['den']
	mu = plan['mu']

	# Pad image (if required):	
	im = padImage(im, n_pad0, n_pad1) 

	# (Un)comment the following two lines to use PyFFTW:
	n_byte_align(im, simd_alignment) 
	im = rfft2(im, threads=nr_threads)			

	# (Un)comment the following line to use NumPy:
	#im = rfft2(im)

	# Apply Paganin's (pre-computed) formula:
	im = im / den
		
	# (Un)comment the following two lines to use PyFFTW:
	n_byte_align(im, simd_alignment)
	im = irfft2(im, threads=nr_threads)
		
	# (Un)comment the following line to use NumPy:
	#im = irfft2(im)
				
	im = im.astype(float32)		
	im = -1 / mu * nplog(im)    		

	# Return cropped output:
	return im[marg0:dim0_o + marg0, marg1:dim1_o + marg1] 

