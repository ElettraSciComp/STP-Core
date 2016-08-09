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

from numpy import uint16, float32, iinfo, finfo, copy, zeros, median, float32


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


def rivers(im, args):
    """Process a sinogram image with the Rivers de-striping algorithm.

    Parameters
    ----------
    im : array_like
        Image data as numpy array.

    n : int
        Size of the median filtering.

	(Parameter n has to passed as a string ended by ;)
       
    Example (using tiffile.py)
    --------------------------
    >>> im = imread('original.tif')
    >>> im = rivers(im, '13;')    
    >>> imsave('filtered.tif', im) 

    References
    ----------
    M. Rivers, http://cars9.uchicago.edu/software/epics/tomoRecon.html.

    """   
    # Get args:
    param1, param2 = args.split(";")    
    n = int(param1)

    # Compute mean of each column:
    col = im.mean(axis=0)

    # Perform low pass filtering:
    flt_col = _medfilt(col, n)
    
    # Apply compensation on each row:
    for i in range(0, im.shape[0]):
        im[i,] = im[i,] - (col - flt_col)

    # Return image:
    return im.astype(float32)
