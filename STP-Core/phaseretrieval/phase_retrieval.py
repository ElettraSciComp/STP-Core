# #########################################################################
# (C) 2016 Elettra - Sincrotrone Trieste S.C.p.A.. All rights reserved.   #
#                                                                         #
# Copyright 2016. Elettra - Sincrotrone Trieste S.C.p.A. THE COMPANY      #
# ELETTRA - SINCROTRONE TRIESTE S.C.P.A. IS NOT REPONSIBLE FOR THE USE    #
# OF THIS SOFTWARE. If software is modified to produce derivative works,  #
# such modified software should be clearly marked, so as not to confuse   #
# it with the version available from Elettra Sincrotrone Trieste S.C.p.A. #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of Elettra - Sincotrone Trieste S.C.p.A nor      #
#       the names of its contributors may be used to endorse or promote   #
#       products derived from this software without specific prior        #
#       written permission.                                               #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY ELETTRA - SINCROTRONE TRIESTE S.C.P.A. AND #
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,  #
# BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND       #
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL      #
# ELETTRA - SINCROTRONE TRIESTE S.C.P.A. OR CONTRIBUTORS BE LIABLE FOR    #
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL  #
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE       #
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS           #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER    #
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR         #
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF  #
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                              #
# #########################################################################

#
# Author: Francesco Brun
# Last modified: April, 4th 2016
#

from numpy import float32, complex64, arange, ComplexWarning, finfo
from numpy import copy, meshgrid
from numpy import pi, log as nplog, cos as npcos, sin as npsin

# (Un)comment the related lines to use either NumPY or PyFFTW:
#from numpy.fft import fft2, ifft2
from pyfftw import n_byte_align, simd_alignment
from pyfftw.interfaces.numpy_fft import fft2, ifft2
from pyfftw.interfaces.cache import enable as pyfftw_cache_enable, disable as pyfftw_cache_disable
from pyfftw.interfaces.cache import set_keepalive_time as pyfftw_set_keepalive_time
from numpy.fft import fftshift, ifftshift
from warnings import simplefilter

from utils.padding import upperPowerOfTwo, padImage


def prepare_plan(im, beta, delta, energy, distance, pixsize, method=1, padding=False):
	"""Pre-compute data to save time in further execution of phase_retrieval

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
	method : int 
		Phase retrieval algorithm {1 = TIE (default), 2 = Born, 3 = Rytov, 4 = Wu}
	padding : bool
		Apply image padding to better process the boundary of the image
	
	"""
	# Get additional values:
	lam = (12.398424 * 10 ** (-7)) / energy # in mm
	mu = 4 * pi * beta / lam
		
	# Replicate pad image to power-of-2 dimensions:
	dim0_o = im.shape[0]
	dim1_o = im.shape[1]
	if (padding):		
		n_pad0 = dim0_o
		n_pad1 = n_pad1 = im.shape[1] + im.shape[1] / 2
	else:
		n_pad0 = dim0_o
		n_pad1 = dim1_o

	# Set the transformed frequencies according to pixelsize:
	rows = n_pad0
	cols = n_pad1
	ulim = arange(-(cols) / 2, (cols) / 2)
	ulim = ulim * (2 * pi / (cols * pixsize))
	vlim = arange(-(rows) / 2, (rows) / 2)  
	vlim = vlim * (2 * pi / (rows * pixsize))
	u,v = meshgrid(ulim, vlim)

	# Apply formula:
	if method == 1:    
		den = 1 + distance * delta / mu * (u * u + v * v) + finfo(float32).eps # Avoids division by zero		
	elif method == 2:
		chi = pi * lam * distance * (u * u + v * v)
		den = (beta / delta) * npcos(chi) + npsin(chi) + finfo(float32).eps # Avoids division by zero		
	elif method == 3:
		chi = pi * lam * distance * (u * u + v * v)
		den = (beta / delta) * npcos(chi) + npsin(chi) + finfo(float32).eps # Avoids division by zero		
	elif method == 4:
		den = 1 + pi * (delta / beta) * lam * distance * (u * u + v * v) + finfo(float32).eps        
		
	# Shift the denominator now:
	den = fftshift(den)

	return {'dim0':dim0_o, 'dim1':dim1_o ,'npad0':n_pad0, 'npad1':n_pad1, 'den':den , 'mu':mu }
	

def phase_retrieval(im, plan, method=1, nr_threads=2):
	"""Process a tomographic projection image with the selected phase retrieval algorithm.

	Parameters
	----------
	im : array_like
		Flat corrected image data as numpy array.
	plan : structure
		Structure with pre-computed data (see prepare_plan function)
	method : int 
		Phase retrieval algorithm {1 = TIE (default), 2 = Born, 3 = Rytov, 4 = Wu}
	nr_threads : int 
		Number of threads to be used in the computation of FFT by PyFFTW
		
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
	im  = padImage(im, n_pad0, n_pad1) 

	# (Un)comment the following two lines to use PyFFTW:
	n_byte_align(im, simd_alignment) 
	im = fft2(im, threads=nr_threads)			

	# (Un)comment the following line to use NumPy:	
	#im = fft2(im)			

	# Apply formula:
	if method == 1:		
		im = im / den
		
		# (Un)comment the following two lines to use PyFFTW:
		n_byte_align(im, simd_alignment)
		im = ifft2(im, threads=nr_threads)
		
		# (Un)comment the following line to use NumPy:
		#im = ifft2(im)
		im = im.astype(complex64)
		 		
		im = real(im)
		im = im.astype(float32)		
		im = -1 / mu * nplog(im)    		

	#
	# WARNING: The following methods are not tested
	#
	elif method == 2:
		im = real(ifft2((im - 1.0) / 2.0) / den)
	elif method == 3:
		im = real(ifft2(nplog(im) / 2.0) / den)
	elif method == 4:       
		im = real(ifft2(im / den))
		im = -1 / 2 * (delta / beta) * nplog(im)    

	# Return cropped output:
	return im[marg0:dim0_o + marg0, marg1:dim1_o + marg1]   
