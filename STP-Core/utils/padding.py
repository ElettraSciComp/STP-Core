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

from numpy import zeros, tile, concatenate, ones, linspace, sqrt, finfo, float32, hanning
from ctypes import *

from tifffile import imread, imsave # only for debug

def upperPowerOfTwo(v):
	"""Return the upper power of two of input value

	Parameters
	----------
	v : int
		A positive integer value

	Return value
	----------
	An integer value	
	
	"""
	v = c_uint32(v).value
	v -= 1
	v |= v >> 1
	v |= v >> 2
	v |= v >> 4
	v |= v >> 8
	v |= v >> 16
	v += 1
	return c_int32(v).value

def replicatePadImage(im, marg0, marg1):
	"""Pad the input image by replicating first and last column as well as first and last row
	   the specified number of times.

	Parameters
	----------    
	im : array_like
		Image data as numpy array.

	marg0 : int
		The number of times first and last row have to be replicated.

	marg1 : int
		The number of times first and last column have to be replicated.

	Return value
	----------
	A replicated-padded image.	
	
	"""
	# Pad first side (replicate first column):
	tmp = im[:,0] # Get first column
	tmp = tile(tmp, (marg1,1) ) # Replicate the first column the right number of times
	im  = concatenate( (tmp.T,im), axis=1) # Concatenate tmp before the image

	# Pad second side (replicate last column):
	tmp = im[:,im.shape[1]-1] # Get last column
	tmp = tile(tmp, (marg1,1)) # Replicate the last column the right number of times
	im  = concatenate( (im,tmp.T), axis=1) # Concatenate tmp after the image

	# Pad third side (replicate first row):
	tmp = im[0,:] # Get first row
	tmp = tile(tmp, (marg0,1)) # Replicate the first row the right number of times
	im = concatenate( (tmp,im), axis=0) # Concatenate tmp before the image

	# Pad fourth side (replicate last row):
	tmp = im[im.shape[0]-1,:] # Get last row:
	tmp = tile(tmp, (marg0,1)) # Create a tmp matrix replicating the last row the right number of times
	im = concatenate( (im,tmp), axis=0) # Concatenate tmp after the image

	return im
	
def zeroPadImage(im, marg0, marg1):
	"""Pad the input image by adding zeros.

	Parameters
	----------    
	im : array_like
		Image data as numpy array.

	marg0 : int
		The number of zero rows to add before first and after last row.

	marg1 : int
		The number of zero rows to add before first and after last column.

	Return value
	----------
	A zero-padded image.
	
	"""
	# Pad first side (zeros before first column):
	tmp = zeros(im[:,0].shape) # Get a column of zeros
	tmp = tile(tmp, (marg1,1) ) # Replicate the column the right number of times
	im  = concatenate( (tmp.T,im), axis=1) # Concatenate tmp before the image

	# Pad second side (zeros after last column):
	tmp = zeros(im[:,im.shape[1]-1].shape) # Get a column of zeros
	tmp = tile(tmp, (marg1,1)) # Replicate the column the right number of times
	im  = concatenate( (im,tmp.T), axis=1) # Concatenate tmp after the image

	# Pad third side (zeros before first row):
	tmp = zeros(im[0,:].shape) # Get a row of zeros
	tmp = tile(tmp, (marg0,1)) # Replicate the row the right number of times
	im = concatenate( (tmp,im), axis=0) # Concatenate tmp before the image

	# Pad fourth side (zeros after last row):
	tmp = zeros(im[im.shape[0]-1,:].shape) # Get a row of zeros:
	tmp = tile(tmp, (marg0,1)) # Create a tmp matrix replicating the row the right number of times
	im = concatenate( (im,tmp), axis=0) # Concatenate tmp after the image

	return im

def padImage(im, n_pad0, n_pad1):
	"""Replicate pad the input image to the specified new dimensions.

	Parameters
	----------    
	im : array_like
		Image data as numpy array

	n_pad0 : int
		The new height of the image

	n_pad1 : int
		The new width of the image

	Return value
	----------
	A padded image	
	
	"""
	marg0 = (n_pad0 - im.shape[0]) / 2
	marg0 = marg0 / 2
	marg1 = (n_pad1 - im.shape[1]) / 2
	marg1 = marg1 / 2
	
	# First replicate padding:
	im = replicatePadImage(im, marg0, marg1)
	im = replicatePadImage(im, marg0, marg1)
		
	# Correction for odd/even issues:
	marg0 = n_pad0 - im.shape[0]
	marg1 = n_pad1 - im.shape[1] 
		
	# Now zero padding:
	tmp = zeros(im[im.shape[0]-1,:].shape) # Get last row:
	tmp = tile(tmp, (marg0,1)) # Create a tmp matrix replicating the last row the right number of times
	im = concatenate( (im,tmp), axis=0) # Concatenate tmp after the image
	
	tmp = zeros(im[:,im.shape[1]-1].shape) # Get last column
	tmp = tile(tmp, (marg1,1)) # Replicate the last column the right number of times
	im  = concatenate( (im,tmp.T), axis=1) # Concatenate tmp after the image
			
	return im

def padSmoothWidth(im, n_pad):
	"""Pad the input image to the specified new width by replicate padding with Hanning smoothing to zero.

	Parameters
	----------    
	im : array_like
		Image data as numpy array.

	n_pad : int
		The new width of the image.

	Return value
	----------
	A padded image	
	
	"""
	# Get margin:
	marg = (n_pad - im.shape[1]) / 2
	
	# First replicate padding:
	im = replicatePadImage(im, 0, marg)

	# Prepare smoothing matrix to smooth to zero with a windowing function:
	vscale = ones(im.shape[1] - marg)

	hann   = hanning(marg)
	vleft  = hann[0:marg/2]
	vright = hann[marg/2:]
		
	vrow = concatenate( (vleft,vscale), axis=1)
	vrow = concatenate( (vrow,vright), axis=1)
	vmatrix = tile(vrow, (im.shape[0],1))

	# Correction for odd/even issues:
	marg = im.shape[1] - vmatrix.shape[1]
	tmp = zeros(vmatrix[:,vmatrix.shape[1]-1].shape) # Get last column
	tmp = tile(tmp, (marg,1)) # Replicate the last column the right number of times
	vmatrix = concatenate( (vmatrix,tmp.T), axis=1) # Concatenate tmp after the image

	# Apply smoothing:	
	im = im * vmatrix
	im = im.astype(float32)
	im [im < finfo(float32).eps] = finfo(float32).eps
			
	return im.astype(float32)