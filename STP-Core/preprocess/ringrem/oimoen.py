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
# Last modified: May, 24th 2016
#

from numpy import uint16, float32, iinfo, finfo, ndarray
from numpy import copy

from _medfilt import _medfilt

def oimoen(im, args):
    """Process a sinogram image with the Oimoen de-striping algorithm.

    Parameters
    ----------
    im : array_like
        Image data as numpy array.

    n1 : int
        Size of the median radial filtering.

    n2 : int
        Size of the median azimutal filtering.
       
    Example (using tifffile.py)
    --------------------------
    >>> im = imread('sino_orig.tif')
    >>> im = oimoen(im, 51, 51)    
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

    im1 = im.copy()

    # Radial median filtering:
    for i in range(0, im.shape[0]):

        im1[i,:] = _medfilt(im[i,:], n1)        

    # Create difference image (high-pass filter):
    diff = im - im1

    # Azimutal filtering:
    for i in range(0, im.shape[1]):

        diff[:,i] = _medfilt(diff[:,i], n2)

    # Compensate output image:
    im = im - diff

    # Return image according to input type:
    if (im.dtype == 'uint16'):

        # Check extrema for uint16 images:
        im[im < iinfo(uint16).min] = iinfo(uint16).min
        im[im > iinfo(uint16).max] = iinfo(uint16).max

        # Return image:
        return im.astype(uint16)

    else:

        return im