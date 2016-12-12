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


def usecorrelation( im1, im2 ):
	"""Assess the offset (to be used for e.g. the assessment of the center of rotation or the 
	ovarlap) by computation the peak of the correlation between the two input images.

    Parameters
    ----------    
    im1 : array_like
		Image data as numpy array.

	im2 : array_like
		Image data as numpy array.
			
	Returns
	-------
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


