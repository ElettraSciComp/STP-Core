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
#     * Neither the name of Elettra - Sincrotrone Trieste S.C.p.A nor     #
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
    if (im.dtype == 'uint16'):

        # Check extrema for uint16 images:
        im[im < iinfo(uint16).min] = iinfo(uint16).min
        im[im > iinfo(uint16).max] = iinfo(uint16).max

        # Return image:
        return im.astype(uint16)

    else:

        return im