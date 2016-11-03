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

from numpy.fft import fftshift, ifftshift
from numpy import real, exp as npexp, ComplexWarning
from numpy import arange, floor, kron, ones, float32
from numpy import pad, hanning, zeros, concatenate, tile, finfo

from numpy import zeros, mean, median, var, copy, vstack, float32 # sijbers

from pyfftw import n_byte_align, simd_alignment
#from numpy.fft import fft, ifft
from pyfftw.interfaces.numpy_fft import fft, ifft

from pywt import wavedec2, waverec2

from warnings import simplefilter

from tifffile import imsave, imread


def _sijberspostnov(im, args):
	"""Process a sinogram image with the Sijbers and Postnov de-striping algorithm.

	Parameters
	----------
	im : array_like
		Image data as numpy array. 
   
	winsize : int
		Size of the local floating window used to look for homogeneity.

	thresh : float
		Image rows (within the floating window) having variance below 
		this tresh will be corrected.
	   
	Example
	--------------------------
	>>> im = imread('original.tif')
	>>> im = sijberspostnov_filter(im, 51, 0.001)    
	>>> imsave('filtered.tif', im) 

	References
	----------
	J. Sijbers and A. Postnov, Reduction of ring artifacts in high resolution
	micro-CT reconstructions, Physics in Medicine and Biology 49(14):247-253, 2004.

	"""  

	# Initializations:
	dimx = im.shape[1]
	dimy = im.shape[0]
	
	 # Get args:
	winsize, thresh  = args.split(";")     
	winsize = int(winsize)
	thresh  = float(thresh)

	# Normalize thresh parameter:
	#thresh = thresh*65536.0
		
	glob_art = zeros(dimx)
	prevsize = 0

	# Within a sliding window:
	for i in range(0, dimx - winsize):
			
		ct = 0
		matrix = zeros(winsize)

		# For each line of the current window:
		for j in range(0, dimy):
		
			# Compute the variance within current sliding window:
			v = im[j, i:(i + winsize)]            
			curr_var = var(v)
		
			# If variance is below threshold:
			if (curr_var < thresh): 
				# Add current line with mean subtracted to a temporary matrix:
				v = v - mean(v)   
				matrix = vstack([matrix,v])    
				ct = ct + 1
			
		# Determine local artifact correction vector:
		if (ct > 1):
			ct = ct - 1
			matrix = matrix[1:ct,:]
			loc_art = median(matrix, axis=0)
		else:
			if (ct == 1):
				loc_art = matrix[1,:]
			else:
				loc_art = zeros(winsize)
		   
		# Determine global artifact correction vector:
		for k in range(0, winsize):
			if (matrix.shape[0] > prevsize):
				glob_art[k + i] = loc_art[k]             
	
		prevsize = matrix.shape[0]    

	# Correct each line of the input image:
	for i in range(0, im.shape[0]):
		im[i,:] = im[i,:] - glob_art

	 # Return image:
	return im.astype(float32)


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

def muenchetal(im, args):
	"""Process a sinogram image with the Munch et al. de-striping algorithm.

	Parameters
	----------
	im : array_like
		Image data as numpy array.

	wlevel : int
		Levels of the wavelet decomposition.

	sigma : float
		Smoothing effect.

	(Parameters wlevel and sigma have to passed as a string separated by ;)
	   
	Example (using tiffile.py)
	--------------------------
	>>> im = imread('sino_orig.tif')
	>>> im = munchetal(im, '4;1.0')    
	>>> imsave('sino_flt.tif', im) 

	References
	----------
	B. Munch, P. Trtik, F. Marone, M. Stampanoni, Stripe and ring artifact removal with
	combined wavelet-Fourier filtering, Optics Express 17(10):8567-8591, 2009.

	"""  
	# Disable a warning:
	simplefilter("ignore", ComplexWarning)

	# Get args:
	wlevel, sigma = args.split(";")    
	wlevel = int(wlevel)
	sigma  = float(sigma)

	# The wavelet transform to use : {'haar', 'db1'-'db20', 'sym2'-'sym20', 'coif1'-'coif5', 'dmey'}
	wname = "db5"

	# Wavelet decomposition:
	coeffs = wavedec2(im.astype(float32), wname, level=wlevel)
	coeffsFlt = [coeffs[0]] 

	# FFT transform of horizontal frequency bands:
	for i in range(1, wlevel + 1):  

		# Padding and windowing of input signal:
		n_byte_align(coeffs[i][1], simd_alignment) 
		siz = coeffs[i][1].shape
		tmp = pad(coeffs[i][1], pad_width=((coeffs[i][1].shape[0] / 2, coeffs[i][1].shape[0] / 2), (0,0)), mode='constant') # or 'constant' for zero padding
		tmp = pad(tmp, pad_width=((0,0) ,(coeffs[i][1].shape[1] / 2, coeffs[i][1].shape[1] / 2)), mode='constant')    # or 'constant' for zero padding
		tmp = _windowing_lr(tmp, siz[1])
		tmp = _windowing_lr(tmp.T, siz[0]).T	

		###tmp = coeffs[i][1]		

		# FFT:
		fcV = fftshift(fft(tmp, axis=0, threads=2))  
		my, mx = fcV.shape
		
		# Damping of vertical stripes:
		damp = 1 - npexp(-(arange(-floor(my / 2.),-floor(my / 2.) + my) ** 2) / (2 * (sigma ** 2)))      
		dampprime = kron(ones((1,mx)), damp.reshape((damp.shape[0],1)))
		fcV = fcV * dampprime    

		# Inverse FFT:
		fcV = ifftshift(fcV)
		n_byte_align(fcV, simd_alignment)
		fcVflt = ifft(fcV, axis=0, threads=2)

		## Crop image:
		tmp = fcVflt[fcVflt.shape[0] / 4:(fcVflt.shape[0] / 4 + siz[0]), fcVflt.shape[1] / 4:(fcVflt.shape[1] / 4 + siz[1])]
		####tmp= fcVflt

		# Dump back coefficients:
		cVHDtup = (coeffs[i][0], tmp, coeffs[i][2])          
		coeffsFlt.append(cVHDtup)

	# Get wavelet reconstruction:
	im_f = real(waverec2(coeffsFlt, wname))

	# Return filtered image (an additional row and/or column might be present):
	return im_f[0:im.shape[0],0:im.shape[1]].astype(float32)