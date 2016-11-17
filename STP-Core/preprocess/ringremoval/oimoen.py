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

from numpy import uint16, float32, iinfo, finfo, ndarray
from numpy import copy, pad, zeros, median


def _medfilt (x, k):
	"""Apply a length-k median filter to a 1D array x.
	Boundaries are extended by repeating endpoints.

	Code from https://gist.github.com/bhawkins/3535131

	"""	
	k2 = (k - 1) // 2
	y = zeros ((len (x), k), dtype=x.dtype)
	y[:,k2] = x
	for i in range (k2):
		j = k2 - i
		y[j:,i] = x[:-j]
		y[:j,i] = x[0]
		y[:-j,-(i+1)] = x[j:]
		y[-j:,-(i+1)] = x[-1]
		
	return median (y, axis=1)


def oimoen(im, args):
    """Process a sinogram image with the Oimoen de-striping algorithm.

    Parameters
    ----------
    im : array_like
        Image data as numpy array.

    n1 : int
        Size of the horizontal filtering.

    n2 : int
        Size of the vertical filtering.

	(Parameters n1 and n2 have to passed as a string separated by ;)
       
    Example (using tifffile.py)
    --------------------------
    >>> im = imread('sino_orig.tif')
    >>> im = oimoen(im, '51;51')    
    >>> imsave('sino_flt.tif', im) 

    References
    ----------
    M.J. Oimoen, An effective filter for removal of production artifacts in U.S. 
    geological survey 7.5-minute digital elevation models, Proc. of the 14th Int. 
    Conf. on Applied Geologic Remote Sensing, Las Vegas, Nevada, 6-8 November, 
    2000, pp. 311-319.

    """    
    # Get args:
    param1, param2 = args.split(";")    
    n1 = int(param1) 
    n2 = int(param2)

	# Padding:
    im = pad(im, ((n2 + n1, n2 + n1), (0, 0)), 'symmetric')
    im = pad(im, ((0, 0), (n1 + n2, n1 + n2)), 'edge')

    im1 = im.copy()

    # Horizontal median filtering:
    for i in range(0, im1.shape[0]):

        im1[i,:] = _medfilt(im1[i,:], n1)        

    # Create difference image (high-pass filter):
    diff = im - im1

    # Vertical filtering:
    for i in range(0, diff.shape[1]):

        diff[:,i] = _medfilt(diff[:,i], n2)

    # Compensate output image:
    im = im - diff

	# Crop padded image:
    im = im[(n2 + n1):im.shape[0] - (n1 + n2), (n1 + n2):im.shape[1] - (n1 + n2)]	

    # Return image according to input type:
    return im.astype(float32)