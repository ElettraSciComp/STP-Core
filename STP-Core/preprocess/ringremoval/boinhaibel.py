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

from numpy import uint16, float32, iinfo, finfo, ndarray, copy, zeros, median


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


def boinhaibel(im, args):
    """Process a sinogram image with the Boin and Haibel de-striping algorithm.

    Parameters
    ----------
    im : array_like
        Image data as numpy array.

    n : int
        Size of the median filtering.

	(Parameter n has to passed as a string ended by ;)
       
    Example (using tifffile.py)
    --------------------------
    >>> im = imread('sino_orig.tif')
    >>> im = boinhaibel(im, '11;')    
    >>> imsave('sino_flt.tif', im) 

    References
    ----------
    M. Boin and A. Haibel, Compensation of ring artefacts in synchrotron 
    tomographic images, Optics Express 14(25):12071-12075, 2006.

    """    
    # Get args:
    param1, param2 = args.split(";")    
    n = int(param1)

    # Compute sum of each column (avoid further division by zero):
    col = im.sum(axis=0) + finfo(float32).eps

    # Perform low pass filtering:
    flt_col = _medfilt(col, n)

    # Apply compensation on each row:
    for i in range(0, im.shape[0]):
        im[i,] = im[i,] * (flt_col / col)

    # Return image:
    return im.astype(float32)