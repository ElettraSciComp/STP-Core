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

from numpy import uint16, float32, iinfo, finfo, ndarray
from numpy import real, copy
from numpy.fft import fft2, ifft2
#from pyfftw.interfaces.numpy_fft import fft2, ifft2

from _medfilt import _medfilt

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