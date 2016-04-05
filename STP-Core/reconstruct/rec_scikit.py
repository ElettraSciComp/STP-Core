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

from numpy import float32, double, linspace, pi, zeros
from skimage.transform import iradon, iradon_sart


def recon_scikit_sart(im, angles, iter):
	"""Reconstruct a sinogram with the SART algorithm of the Scikit-Image Python package

	Parameters
	----------
	im : array_like
		Sinogram image data as numpy array.

	angles : double
		Value in radians representing the number of angles of the input sinogram.

	iterations : int
		Number of iterations for the SART technique.
	
	"""

	im = im.astype(double)	
	iter = int(iter)

	# Actual reconstruction with SART:
	rec = zeros((im.shape[1], im.shape[1]))
	for i in range(0, iter):      
		rec = iradon_sart(im.T, linspace(0, angles / pi * 180.0, im.shape[0]), rec)
	
	return rec.astype(float32)
	
def recon_scikit_fbp(im, angles, filter_type):		
	"""Reconstruct a sinogram with the FBP algorithm of the Scikit-Image Python package

	Parameters
	----------
	im : array_like
		Sinogram image data as numpy array.

	angles : double
		Value in radians representing the number of angles of the input sinogram.

	filter_type : string
		A string with e.g. "ramp", "shepp-logan", "cosine", "hamming", "hann".
	
	"""
	# Filters: ramp, shepp-logan, cosine, hamming, hann
	# Interp: ‘linear’, ‘nearest’, and ‘cubic’
	rec = iradon(im.T, linspace(0, angles / pi * 180.0, im.shape[0], False), im.shape[1] , filter_type, 'nearest', True)
	
	return rec