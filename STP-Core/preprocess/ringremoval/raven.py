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
from numpy import real, copy, zeros, median
from numpy.fft import fft2, ifft2
#from pyfftw.interfaces.numpy_fft import fft2, ifft2


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


def raven(im, args):
    """Process a sinogram image with the Raven de-striping algorithm.
	
	A median filter is proposed rather than the Butterworth filter 
	proposed in the original article.

    Parameters
    ----------
    im : array_like
        Image data as numpy array.

    n : int
        Size of the median filtering.
           
    Example (using tiffile.py)
    --------------------------
    >>> im = imread('sino_orig.tif')
    >>> im = raven(im, 11)    
    >>> imsave('sino_flt.tif', im) 

    References
    ----------
    C. Raven, Numerical removal of ring artifacts in microtomography,
    Review of Scientific Instruments 69(8):2978-2980, 1998.

    """    
    # Get args:
    param1, param2 = args.split(";")    
    n = int(param1)
    
    # Compute FT:
    im = fft2(im) 

    # Median filter: 
    # tmp   = concatenate((im[0:3,:], im[-2:,:]), axis=0)
    # im[0,:] = numpy.median(tmp, axis=0);
    im[0,:] = _medfilt(im[0,:], n)

    # Compute inverse FFT of the filtered data:
    im = real(ifft2(im))

    # Return image according to input type:
    if (im.dtype == 'uint16'):
        
        # Check extrema for uint16 images:
        im[im < iinfo(uint16).min] = iinfo(uint16).min
        im[im > iinfo(uint16).max] = iinfo(uint16).max

        # Return image:
        return im.astype(uint16)
    else:
        return im