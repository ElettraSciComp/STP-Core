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
       
    Example (using tiffile.py)
    --------------------------
    >>> im = imread('sino_orig.tif')
    >>> im = munchetal(im, 4, 1.0)    
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
    wname = "db2"

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

    # Return image according to input type:
    if (im.dtype == 'uint16'):
        
        # Check extrema for uint16 images:
        im_f[im_f < iinfo(uint16).min] = iinfo(uint16).min
        im_f[im_f > iinfo(uint16).max] = iinfo(uint16).max

        # Return filtered image (an additional row and/or column might be
        # present):
        return im_f[0:im.shape[0],0:im.shape[1]].astype(uint16)
    else:
        return im_f[0:im.shape[0],0:im.shape[1]]