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

from numpy import float32, double, finfo, argmax, unravel_index, pi, log
from numpy import zeros, ones, ceil, round, clip, sort, fliplr, arange, roll
from numpy import floor, linspace, sum, abs, argmin, int16, vstack, real

# (Un)comment the related lines to use either NumPY or PyFFTW:
#from numpy.fft import fft2, ifft2
from numpy.fft import fftshift
from pyfftw import n_byte_align, simd_alignment
from pyfftw.interfaces.numpy_fft import fft2, ifft2
from pyfftw.interfaces.cache import enable as pyfftw_cache_enable, disable as pyfftw_cache_disable
from pyfftw.interfaces.cache import set_keepalive_time as pyfftw_set_keepalive_time

from tifffile import imread, imsave # only for debug



def usecorrelation( im1, im2 ):
	"""Assess the offset (to be used for e.g. the assessment of the center of rotation or the 
	ovarlap) by computation the peak of the correlation between the two input images.

    Parameters
    ----------    
    im1 : array_like
		Image data as numpy array.

	im2 : array_like
		Image data as numpy array.
			
	Return value
	----------
	An integer value of the location of the maximum peak correlation.
	
    """
	# Fourier transform both images:
	f_im1 = fft2( im1.astype(float32), threads=2 );
	f_im2 = fft2( im2.astype(float32), threads=2 );

	# Perform phase correlation (amplitude is normalized):
	fc    = f_im1 * ( f_im2.conjugate() );
	fcn   = fc / abs(fc);

	# Inverse fourier of peak correlation matrix and max location:
	peak_correlation_matrix = real( ifft2( fcn, threads=2 ));

	# Calculate actual translation:
	max_ix = argmax(peak_correlation_matrix.flatten())
	(row, col) = unravel_index(max_ix, peak_correlation_matrix.shape)
	
	if ( col < (peak_correlation_matrix.shape[1]/2) ):
		col = - (col - 1);
	else:
		col = peak_correlation_matrix.shape[1] - (col - 1);	

	return col / 2;


