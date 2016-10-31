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
# Last modified: Setp, 28th 2016
#

from numpy import float32, finfo, ndarray
from numpy import median, amin, amax, nonzero
from numpy import tile, concatenate, reshape, interp

from scipy.ndimage.filters import median_filter

def _dead_correction (im):
	"""Correct dead pixels by inteporlation

	Parameters
	----------
	im : array_like
		Image data (sinogram) as numpy array. 
	
	"""
	# Compensate with line-by-line interpolation (better suited for large dead areas):
	im[ im < 0.0 ] = 0.0
	im_f = im.flatten()
	val, x = (im_f == 0), lambda z: z.nonzero()[0]
	im_f[val] = interp(x(val), x(~val), im_f[~val])
	im = reshape(im_f, (im.shape[1], im.shape[0]), order='F').copy().T

	return im 

def _afterglow_correction (im):
	"""Correct dead pixels by adaptive median filtering

	Parameters
	----------
	im : array_like
		Image data (sinogram) as numpy array. 
	
	"""

	# Quick and dirty compensation for detector afterglow (it works well for isolated spots):
	size_ct = 3
	while ( ( float(amin(im)) <  0.0) and (size_ct <= 7) ):			
		im_f = median_filter(im, size_ct)
		im[im < 0.0] = im_f[im < 0.0]					
		size_ct += 2

	# Compensate negative values by replacing them with the average value of the image:		
	if (float(amin(im)) <  finfo(float32).eps):			
		rplc_value = sum(im [im > finfo(float32).eps]) / sum(im > finfo(float32).eps)		
		im [im < finfo(float32).eps] = rplc_value

	return im


def flat_fielding (im, i, plan, flat_end, half_half, half_half_line, norm_sx, norm_dx):
	"""Process a sinogram with conventional flat fielding plus reference normalization.

	Parameters
	----------
	im : array_like
		Image data as numpy array
	i : int
		Index of the sinogram with reference to the height of a projection
	plan : structure
		Structure created by the extract_flatdark function (see extract_flatdark.py). 
		This structure contains the flat/dark images acquired before the acquisition of 
		the projections and the flat/dark images acquired after the acquisition of the
		projections as well as a few flags.
	flat_end : bool
		True if the process considers the flat/dark images (if any) acquired after the 
		acquisition of the projections.
	half_half : bool
		True if the process has to be separated by processing the first part of the
		sinogram with the flat/dark images acquired before the acquisition of the 
		projections and the second part with the flat/dark images acquired after the 
		acquisition of the projections.
	half_half_line : int
		Usually this value is equal to the height of the projection FOV / 2 but the two parts 
		of the sinogram to process can have a different size.
	norm_sx : int
		Width in pixels of the left window to be consider for the normalization of the 
		sinogram. This value has to be zero in the case of ROI-CT.
	norm_dx : int
		Width in pixels of the right window to be consider for the normalization of the 
		sinogram. This value has to be zero in the case of ROI-CT.
	   
	Example (using h5py, tdf.py, tifffile.py)
	--------------------------
	>>> sino_idx = 512
	>>> f    = getHDF5('dataset.h5', 'r')
	>>> im   = tdf.read_sino(f['exchange/data'], sino_idx)
	>>> plan = extract_flatdark(f_in, True, False, False, 'tomo', 'dark', 'flat', 'logfile.txt') 
	>>> im   = flat_fielding(im, sino_idx, plan, True, True, 900, 0, 0)  
	>>> imsave('sino_corr.tif', im) 

	"""    
	
	try:

		# Extract plan values:
		im_flat = plan['im_flat']
		im_dark = plan['im_dark']
		im_flat_after = plan['im_flat_after']	
		im_dark_after = plan['im_dark_after']

		skip_flat = plan['skip_flat']
		skip_flat_after = plan['skip_flat_after']
	
		if not isinstance(im_dark, ndarray):
			im_dark = im_dark_after
	
		if not isinstance(im_flat, ndarray):
			im_flat = im_flat_after
			
		# Flat correct the image to process:
		if not skip_flat:
					
			im_flat_curr = im_flat
			im_dark_curr = im_dark
			if flat_end and not skip_flat_after and half_half:
				# Half-and-half mode
				if ((norm_sx == 0) and (norm_dx == 0)):
					norm_coeff = 1.0
				else:
					# Get the air image:
					if (norm_dx == 0):
						im_air = im[:,0:norm_sx]							
					else:
						im_sx = im[:,0:norm_sx]							
						im_dx = im[:,-norm_dx:]					
						im_air = concatenate((im_sx,im_dx), axis=1)	
						
					# Get only the i-th row from flat and dark:					
					if (norm_dx == 0):
						im_flat_air_before = im_flat[i,0:norm_sx]
						im_flat_air_after = im_flat_after[i,0:norm_sx]						
					else:	
						im_flat_sx_before = im_flat[i,0:norm_sx]						
						im_flat_dx_before = im_flat[i,-norm_dx:]					
						im_flat_air_before = concatenate((im_flat_sx_before,im_flat_dx_before), axis=1)
						im_flat_sx_after = im_flat_after[i,0:norm_sx]								
						im_flat_dx_after = im_flat_after[i,-norm_dx:]					
						im_flat_air_after = concatenate((im_flat_sx_after,im_flat_dx_after), axis=1)		
						
					im_flat_air_before = tile(im_flat_air_before, (half_half_line,1)) 
					im_flat_air_after = tile(im_flat_air_after, (im.shape[0]-half_half_line,1)) 
					im_flat_air = concatenate((im_flat_air_before,im_flat_air_after), axis=0)					
					
					if (norm_dx == 0):
						im_dark_air_before = im_dark[i,0:norm_sx]
						im_dark_air_after = im_dark_after[i,0:norm_sx]	
					else:
						im_dark_sx_before = im_dark[i,0:norm_sx]								
						im_dark_dx_before = im_dark[i,-norm_dx:]
						im_dark_air_before = concatenate((im_dark_sx_before,im_dark_dx_before), axis=1)
						im_dark_sx_after = im_dark_after[i,0:norm_sx]								
						im_dark_dx_after = im_dark_after[i,-norm_dx:]					
						im_dark_air_after = concatenate((im_dark_sx_after,im_dark_dx_after), axis=1)					
					
					im_dark_air_before = tile(im_dark_air_before, (half_half_line,1)) 
					im_dark_air_after = tile(im_dark_air_after, (im.shape[0]-half_half_line,1)) 
					im_dark_air = concatenate((im_dark_air_before,im_dark_air_after), axis=0)		
								
					# Set a norm coefficient for avoiding for cycle:
					norm_coeff = median(im_air, axis=1) / (median(im_flat_air, axis=1) + finfo(float32).eps)
					norm_coeff = tile(norm_coeff, (im.shape[1],1)) 
					norm_coeff = norm_coeff.T
						
				# Create flat and dark images replicating the i-th row the proper number of times:
				tmp_flat_before = tile(im_flat[i,:], (half_half_line,1)) 
				tmp_flat_after = tile(im_flat_after[i,:], (im.shape[0]-half_half_line,1)) 
				tmp_flat = concatenate((tmp_flat_before,tmp_flat_after), axis=0)				
				tmp_dark_before = tile(im_dark[i,:], (half_half_line,1)) 
				tmp_dark_after = tile(im_dark_after[i,:], (im.shape[0]-half_half_line,1)) 
				tmp_dark = concatenate((tmp_dark_before,tmp_dark_after), axis=0)
			
			else:
				# The same flat for all the images:				
				if flat_end and not skip_flat_after:
					# Use the ones acquired after the projections:
					im_flat = im_flat_after
					im_dark = im_dark_after

				if ((norm_sx == 0) and (norm_dx == 0)):
					norm_coeff = 1.0								
				else:					
					# Get the air image:
					if (norm_dx == 0):
						im_air = im[:,0:norm_sx]							
					else:
						im_sx = im[:,0:norm_sx]							
						im_dx = im[:,-norm_dx:]
						im_air = concatenate((im_sx,im_dx), axis=1)		
						
					# Get only the i-th row from flat and dark:
					if (norm_dx == 0):
						im_flat_air = im_flat[i,0:norm_sx]	
					else:
						im_flat_sx = im_flat[i,0:norm_sx]								
						im_flat_dx = im_flat[i,-norm_dx:]
						im_flat_air = concatenate((im_flat_sx,im_flat_dx), axis=1)	
						
					im_flat_air = tile(im_flat_air, (im.shape[0],1)) 
					
					if (norm_dx == 0):
						im_dark_air = im_dark[i,0:norm_sx]								
					else:
						im_dark_sx = im_dark[i,0:norm_sx]								
						im_dark_dx = im_dark[i,-norm_dx:]
						im_dark_air = concatenate((im_dark_sx,im_dark_dx), axis=1)						
						
					im_dark_air = tile(im_dark_air, (im.shape[0],1)) 	
								
					# Set a norm coefficient for avoiding for cycle:
					norm_coeff = median(im_air, axis=1) / (median(im_flat_air, axis=1) + finfo(float32).eps)
					norm_coeff = tile(norm_coeff, (im.shape[1],1)) 
					norm_coeff = norm_coeff.T	

				# Create flat and dark images replicating the i-th row the proper number of times:
				tmp_flat = tile(im_flat[i,:], (im.shape[0],1)) 
				tmp_dark = tile(im_dark[i,:], (im.shape[0],1)) 								

			# Dead pixel correction:
			im = _dead_correction(im.astype(float32))
			tmp_flat = _dead_correction(tmp_flat.astype(float32))
						
			# Do actual flat fielding:
			im = ((im - tmp_dark) / ((tmp_flat - tmp_dark) * norm_coeff + finfo(float32).eps)).astype(float32)	
	
			# Correct for afteglow:
			im = _afterglow_correction(im.astype(float32))

	finally:

		# Return pre-processed image:
		return im.astype(float32)