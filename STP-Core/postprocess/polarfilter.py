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
# Last modified: April, 6th 2017
#

from numpy import float32, amin, amax, sqrt, max, pad

import imp, inspect, os
import polarfilters
import cv2

def polarfilter(im, polarfilt_opt):	
	"""Post-process a reconstructed image with a filter in polar coordinates.

	Parameters
	----------
	im : array_like
		Image data as numpy array. 

	filt_opt : string
		String containing filter method and the related parameters.

	"""
	# Get method and args:
	method, args = polarfilt_opt.split(":", 1)

	# The "none" filter means no filtering:
	if (method != "none"):

		# Dinamically load module:
		path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
		str = os.path.join(path, "polarfilters",  method + '.py')
		m = imp.load_source(method, str)
			
		# Convert to 8-bit or 16-bit:
		filt_method, filt_args = polarfilt_opt.split(":", 1)			
				
		# Get original size:
		origsize = im.shape

		# Up-scaling:
		im = cv2.resize(im, None, 2, 2, cv2.INTER_CUBIC)
		rows, cols = im.shape
		cen_x = im.shape[1] / 2
		cen_y = im.shape[0] / 2

		# To polar:
		im = cv2.linearPolar(im, (cen_x, cen_y), amax([rows,cols]), cv2.INTER_CUBIC)

		# Padding:
		cropsize = im.shape
		im = pad(im, ((origsize[0] / 4, origsize[0] / 4), (origsize[1] / 2, 0)), 'symmetric')  
		
		# Call the filter dynamically:
		im = getattr(m, method)(im, args)

		# Crop:
		im = im[origsize[0] / 4:origsize[0] / 4 + cropsize[0],origsize[1] / 2:origsize[1] / 2 + cropsize[1]]

		# Back to cartesian:
		im = cv2.linearPolar(im, (cen_x, cen_y), amax([rows,cols]), cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)

		# Down-scaling to original size:
		im = cv2.resize(im, (origsize[0], origsize[1]), interpolation = cv2.INTER_CUBIC)

	return im 
