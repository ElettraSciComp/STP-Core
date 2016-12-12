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
from numpy import copy, meshgrid, real, ceil, floor, concatenate, logical_and
from numpy import sign, fabs, pi, log as nplog, cos as npcos, sin as npsin

# (Un)comment the related lines to use either NumPY or PyFFTW:
#from numpy.fft import fft2, ifft2
from pyfftw import n_byte_align, simd_alignment
from pyfftw.interfaces.numpy_fft import fft2, ifft2
from warnings import simplefilter

from utils.padding import upperPowerOfTwo, padImage


def phrt_plan (im, energy, distance, pixsize, regpar, thresh, method, padding):
	"""Pre-compute data to save time in further execution of phase_retrieval.

	Parameters
	----------
	im : array_like
		Image data as numpy array. Only image size (shape) is actually used.
	
	energy [KeV]: double
		Energy in KeV of the incident X-ray beam.
	
	distance [mm]: double
		Sample-to-detector distance in mm.
	
	pixsize [mm]: double
		Size in mm of the detector element.
	
	regpar: double
		Regularization parameter: RegPar is - log10 of the constant to be added to the denominator
		to regularize the singularity at zero frequency, i.e. 1/sin(x) -> 1/(sin(x)+10^-RegPar). 
		Typical values in the range [2.0, 3.0]. (Suggestion for default: 2.5).
	
	thresh: double
		Parameter for Quasiparticle phase retrieval which defines the width of the rings to be cropped 
		around the zero crossing of the CTF denominator in Fourier space. Typical values in the range
		[0.01, 0.1]. (Suggestion for default: 0.1).
	
	method : int 
		Phase retrieval algorithm {1 = TIE (default), 2 = CTF, 3 = CTF first-half sine, 4 = Quasiparticle, 
		5 = Quasiparticle first half sine}.
	
	padding : bool
		Apply image padding to better process the boundary of the image.

	References
	----------


	Credits
	-------
	Julian Moosmann, KIT (Germany) is acknowledged for this code
	
	"""
	# Adapt input values:
	distance = distance / 1000.0 # Conversion to m
	pixsize = pixsize / 1000.0 # Conversion to m

	# Get additional values:
	lam = 6.62606896e-34*299792458/(energy*1.60217733e-16)
	k	= 2*pi*lam*distance/(pixsize**2)
			
	# Replicate pad image:
	dim0_o = im.shape[0]
	dim1_o = im.shape[1]
	if (padding):		
		n_pad0 = im.shape[0] + im.shape[0] / 2
		n_pad1 = im.shape[1] + im.shape[1] / 2
	else:
		n_pad0 = dim0_o
		n_pad1 = dim1_o

	# Create coordinates grid:
	xi  = concatenate((arange(0, ceil((n_pad1 - 1)/2.0) + 1) , arange(-(floor((n_pad1 - 1)/2.0)),0)), axis=1) / n_pad1 
	eta = concatenate((arange(0, ceil((n_pad0 - 1)/2.0) + 1) , arange(-(floor((n_pad0 - 1)/2.0)),0)), axis=1) / n_pad0

	[u, v] = meshgrid(xi,eta)	
	u      = k*(u*u + v*v)/2.0

	# Filter:
	if method == 1:	# TIE:
		filter = 0.5 / (u + 10**-regpar)
		
	elif method == 2: # CTF:
		v   = npsin(u)
		filter = 0.5 * sign(v) / (fabs(v) + 10**-regpar)

	elif method == 3: # CTF first-half sine:
		v   = npsin(u)
		filter = 0.5 * sign(v) / (fabs(v) + 10**-regpar)
		filter[ u >= pi ] = 0		

	elif method == 4: # Quasiparticle:
		v   = npsin(u);
		filter = 0.5 * sign(v) / (fabs(v) + 10**-regpar)
		filter[ logical_and( u > pi/2.0, fabs(v) < thresh) ] = 0
		
	elif method == 5: # Quasiparticle first half sine:
		v   = npsin(u)
		filter = 0.5 * sign(v) / (fabs(v) + 10**-regpar)
		filter[ logical_and(u > pi/2.0, fabs(v) < thresh) ] = 0
		filter[ u >= pi ] = 0

	elif method == 6: #	Projected CTF (alternative implementation):
		v   = npsin(u)
		filter = 0.5 * sign(v) / (fabs(v) + 10**-regpar)		
		tmp    = sign(filter) / (2*(thresh + 10**-regpar))
		msk    = logical_and( u > pi/2.0, fabs(v) < thresh)
		filter = filter*(1 - msk) + tmp*msk
		#filter[ u >= pi ] = 0
		
	# Restore zero frequency component:
	filter[0,0] = 0.5 *10**regpar
	
	return {'dim0':dim0_o, 'dim1':dim1_o ,'npad0':n_pad0, 'npad1':n_pad1, 'filter':filter }
	

def phrt(im, plan, method=4, nr_threads=2):
	"""Process a tomographic projection image with the selected phase retrieval algorithm.

	Parameters
	----------
	im : array_like
		Flat corrected image data as numpy array.

	plan : structure
		Structure with pre-computed data (see prepare_plan function)

	method : int 
		Phase retrieval filter {1 = TIE (default), 2 = CTF, 3 = CTF first-half sine, 4 = Quasiparticle, 
		5 = Quasiparticle first half sine}.

	nr_threads : int 
		Number of threads to be used in the computation of FFT by PyFFTW

	Credits
	-------
	Julian Moosmann, KIT (Germany) is acknowledged for this code
		
	"""
	# Extract plan values:
	dim0_o = plan['dim0']
	dim1_o = plan['dim1']
	n_pad0 = plan['npad0']
	n_pad1 = plan['npad1']
	marg0  = (n_pad0 - dim0_o) / 2
	marg1  = (n_pad1 - dim1_o) / 2
	filter = plan['filter']	
	
	# Pad image (if required):	
	im  = padImage(im, n_pad0, n_pad1) 

	# (Un)comment the following two lines to use PyFFTW:
	n_byte_align(im, simd_alignment) 
	im = fft2(im - 1, threads=nr_threads)			

	# (Un)comment the following line to use NumPy:	
	#im = fft2(im - 1)			

	# Apply phase retrieval filter:
	im = filter * im   

	# (Un)comment the following two lines to use PyFFTW:
	n_byte_align(im, simd_alignment)
	im = real(ifft2(im, threads=nr_threads))	

	# (Un)comment the following line to use NumPy:	
	#im = real(ifft2(im))	

	# Return the negative:
	im = - im

	# Return cropped output:
	return im[marg0:dim0_o + marg0, marg1:dim1_o + marg1]   
