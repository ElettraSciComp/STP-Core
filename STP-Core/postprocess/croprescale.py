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

from numpy import uint8, uint16

def croprescale( im, convert_opt, crop_opt ):	
	"""Post-process a reconstructed image.

    Parameters
    ----------
    im : array_like
		Image data as numpy array. 
	convert_opt : string
		String containing degradation method (8-bit or 16-bit) and min/max 
		rescaling value (e.g. "linear8:-0.01;0.01"). In current version only
		"linear" for 16-bit and "linear8" are implemented.
	crop_opt : double
		String containing the parameters to crop an image separated by : with
		order top, bottom, left, right. (e.g. "100:100:100:100")	

    """
	# Convert to 8-bit or 16-bit:
	conv_method, conv_args = convert_opt.split(":", 1)			
	if (conv_method.startswith('linear')):		
	
		# Get args:
		min, max = conv_args.split(";")    
		min = float(min)
		max = float(max)
		
		im = (im - min) / (max - min);
		im[ im < 0.0] = 0.0;
		im[ im > 1.0] = 1.0;
		
		if (conv_method == 'linear8'):
			im = im*255.0	
			im = im.astype(uint8)	
		else:
			im = im*65535.0	
			im = im.astype(uint16)		
		
	# Crop the image:	
	crop_opt = crop_opt.split(":")
	
	#im = im[crop_top:im.shape[0]-crop_bottom,crop_left:im.shape[1]-crop_right]	
	im = im[int(crop_opt[0]):im.shape[0]-int(crop_opt[1]),int(crop_opt[2]):im.shape[1]-int(crop_opt[3])]		
	
	return im
