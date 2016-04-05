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
	
	# First replicate padding and then zero padding:
	im = replicatePadImage(im, marg0, marg1)
	im = replicatePadImage(im, marg0, marg1)
		
	# Correction for odd/even issues:
	marg0 = n_pad0 - im.shape[0]
	marg1 = n_pad1 - im.shape[1] 
		
	tmp = zeros(im[im.shape[0]-1,:].shape) # Get last row:
	tmp = tile(tmp, (marg0,1)) # Create a tmp matrix replicating the last row the right number of times
	im = concatenate( (im,tmp), axis=0) # Concatenate tmp after the image
	
	tmp = zeros(im[:,im.shape[1]-1].shape) # Get last column
	tmp = tile(tmp, (marg1,1)) # Replicate the last column the right number of times
	im  = concatenate( (im,tmp.T), axis=1) # Concatenate tmp after the image
			
	return im

def padSino(im, n_pad):
	"""Pad the input sinogram to the specified width by replicate padding with Hanning smoothing to zero.

    Parameters
    ----------    
    im : array_like
		Image data as numpy array.

	n_pad : int
		The new width of the sinogram.

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
	im [im < 0.0] = 0.0
	im = im + finfo(float32).eps	
			
	return im.astype(float32)