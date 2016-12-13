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

from numpy import rot90, float32, copy, linspace 
from scipy.misc import imresize #scipy 0.12

from _gridrec import paralrecon

#from tifffile import *

def recon_gridrec(im1, im2, angles, oversampling):
	"""Reconstruct two sinograms (of the same CT scan) with direct Fourier algorithm.

	Parameters
	----------
	im1 : array_like
		Sinogram image data as numpy array.

	im2 : array_like
		Sinogram image data as numpy array.

	angles : double
		Value in radians representing the number of angles of the input sinogram.

	oversampling : double
		Input sinogram is rescaled to increase the sampling of the Fourier space and
		avoid artifacts. Suggested value in the range [1.2,1.6]. 
	
	"""
	v_angles = linspace(0, angles, im1.shape[0], False).astype(float32)

	# Call C-code for gridrec with oversampling:
	[out1, out2] = paralrecon(im1, im2, v_angles, float(oversampling))  

	# Rescale output (if oversampling used):
	out1 = imresize(out1, (im1.shape[1],im1.shape[1]), interp='bicubic', mode='F')
	out2 = imresize(out2, (im2.shape[1],im2.shape[1]), interp='bicubic', mode='F')

	# Rotate images 90 degrees towards the left:
	out1 = rot90(out1)
	out2 = rot90(out2)

	# Return output:  
	return [out1.astype(float32), out2.astype(float32)]