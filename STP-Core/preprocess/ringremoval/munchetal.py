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

from numpy.fft import fft, ifft, fftshift, ifftshift
from numpy import real, exp as npexp, ComplexWarning
from numpy import arange, floor, kron, ones, float32

#from pyfftw.interfaces.numpy_fft import fft, ifft
from pywt import wavedec2, waverec2

from warnings import simplefilter


def munchetal(im, args):
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
    wname = "db4"

    # Wavelet decomposition:
    coeffs = wavedec2(im.astype(float32), wname, level=wlevel)
    coeffsFlt = [coeffs[0]] 

    # FFT transform of horizontal frequency bands:
    for i in range(1, wlevel + 1):  

        # FFT:
        fcV = fftshift(fft(coeffs[i][1], axis=0))  
        my, mx = fcV.shape
        
        # Damping of vertical stripes:
        damp = 1 - npexp(-(arange(-floor(my / 2.),-floor(my / 2.) + my) ** 2) / (2 * (sigma ** 2)))      
        dampprime = kron(ones((1,mx)), damp.reshape((damp.shape[0],1)))
        fcV = fcV * dampprime    

        # Inverse FFT:
        fcVflt = ifft(ifftshift(fcV), axis=0)
        cVHDtup = (coeffs[i][0], fcVflt, coeffs[i][2])             
        coeffsFlt.append(cVHDtup)

    # Get wavelet reconstruction:
    im_f = real(waverec2(coeffsFlt, wname))

    # Return filtered image (an additional row and/or column might be present):
    return im_f[0:im.shape[0],0:im.shape[1]].astype(float32)