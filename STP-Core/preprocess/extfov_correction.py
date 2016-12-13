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
# Last modified: December, 12th 2016
#

from numpy import median, fliplr, tile, concatenate, float32, finfo

def extfov_correction(im, ext_fov_rot_right, ext_fov_overlap, ext_fov_normalize, ext_fov_average):
	"""Apply sinogram correction for extended FOV acquisition mode

	Parameters
	----------
	im : array_like
		Image data (sinogram) as numpy array.

	ext_fov_rot_right : bool
		True if the extended FOV mode has been performed with rotation center
		shifted to the right, left otherwise.

	ext_fov_overlap : int
		Number of overlapping pixels.        
		
	ext_fov_normalize : bool
		True to apply line-by-line normalization taking into account the overlap area.
		
	ext_fov_average : bool
		True to consider the average of the whole overlap, false to apply basic half-half
		sticking.
	
	"""
	# Force to floating point (an average process is applied):
	im = im.astype(float32)

	# Correction factor for odd/even shape:
	if (im.shape[0] % 2 == 0): 
		corr = 0
	else:
		corr = 1

	# Flip the bottom part of the sinogram:
	im_bottom = im[im.shape[0] / 2:,:]
	im_bottom = fliplr(im_bottom).astype(float32)	
			
	if (ext_fov_rot_right):
		
		im_overlap_1 = im_bottom[:,-ext_fov_overlap:]
		im_overlap_2 = im[0:im.shape[0] / 2 + corr, 0:ext_fov_overlap]		
			
		if (ext_fov_normalize):
			# Use the overlap area for a line-by-line normalization:
			norm_coeff = median(im_overlap_1, axis=1) / (median(im_overlap_2, axis=1) + finfo(float32).eps)
			norm_coeff = tile(norm_coeff, (im_bottom.shape[1],1))
			im_bottom = im_bottom / (norm_coeff.T)
			
		if (ext_fov_average):
			# Concatenate with the average of the overlap area:
			im_overlap = (im_overlap_1 + im_overlap_2) / 2.0
			tmp = concatenate((im_bottom[:,:-ext_fov_overlap], im_overlap), axis=1)	
			im = concatenate((tmp, im[0:im.shape[0] / 2 + corr, ext_fov_overlap:]), axis=1)	
		else:
			# Concatenate the flipped part of the sinogram on the left:								
			im = concatenate((im_bottom[:,:-ext_fov_overlap / 2], im[0:im.shape[0] / 2 + corr,ext_fov_overlap / 2:]), axis=1)
			
	else:
		
		im_overlap_1 = im_bottom[:,0:ext_fov_overlap]
		im_overlap_2 = im[0:im.shape[0] / 2 + corr, -ext_fov_overlap:]
					
		if (ext_fov_normalize):
			# Use the overlap area for a line-by-line normalization:		
			norm_coeff = median(im_overlap_1, axis=1) / (median(im_overlap_2, axis=1) + finfo(float32).eps)
			norm_coeff = tile(norm_coeff, (im_bottom.shape[1],1))
			im_bottom = im_bottom / (norm_coeff.T)

		if (ext_fov_average):
			# Concatenate with the average of the overlap area:
			im_overlap = (im_overlap_1 + im_overlap_2) / 2.0
			tmp = concatenate((im[0:im.shape[0] / 2 + corr, 0:-ext_fov_overlap], im_overlap), axis=1)	
			im = concatenate((tmp, im_bottom[:,ext_fov_overlap:]), axis=1)			
		else:
			# Concatenate the flipped part of the sinogram on the right:											
			im = concatenate((im[0:im.shape[0] / 2 + corr,:-ext_fov_overlap / 2], im_bottom[:,ext_fov_overlap / 2:]), axis=1)				

	return im.astype(float32)