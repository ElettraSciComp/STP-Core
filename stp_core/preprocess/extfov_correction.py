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

from numpy import fliplr, tile, concatenate

def extfov_correction(im, ext_fov, ext_fov_rot_right, ext_fov_overlap):
	"""Apply sinogram correction for extended FOV acquisition mode

    Parameters
    ----------
    im : array_like
        Image data (sinogram) as numpy array.

    ext : bool
        True if the extended FOV mode has been performed.

    ext_fov_rot_right : bool
        True if the extended FOV mode has been performed with rotation center
		shifted to the right, left otherwise.

	ext_fov_overlap : int
		Number of overlapping pixels.        
    
    """
	if (ext_fov):
			
		im_bottom = im[im.shape[0]/2:,:]
		im_bottom = fliplr(im_bottom)	
			
		if (ext_fov_rot_right):
				
			# Flip the bottom part of the sinogram and put it on the left:					
			if (ext_fov_overlap % 2 == 0):
				im_bottom = im_bottom[:,:-ext_fov_overlap/2]
				if (im.shape[0] % 2 == 0):
					im = concatenate((im_bottom, im[0:im.shape[0]/2,ext_fov_overlap/2:]), axis=1)				
				else:
					im = concatenate((im_bottom, im[0:im.shape[0]/2+1,ext_fov_overlap/2:]), axis=1)
			else:
				im_bottom = im_bottom[:,:-ext_fov_overlap/2 + 1]
				if (im.shape[0] % 2 == 0):
					im = concatenate((im_bottom, im[0:im.shape[0]/2,ext_fov_overlap/2:]), axis=1)				
				else:
					im = concatenate((im_bottom, im[0:im.shape[0]/2+1,ext_fov_overlap/2:]), axis=1)				
			
		else:

			# Flip the bottom part of the sinogram and put it on the right:					
			if (ext_fov_overlap % 2 == 0):
				im_bottom = im_bottom[:,ext_fov_overlap/2:]			
				if (im.shape[0] % 2 == 0):						
					im = concatenate((im[0:im.shape[0]/2,:-ext_fov_overlap/2], im_bottom), axis=1)	
				else:
					im = concatenate((im[0:im.shape[0]/2+1,:-ext_fov_overlap/2], im_bottom), axis=1)							
			else:
				im_bottom = im_bottom[:,ext_fov_overlap/2+1:]
				if (im.shape[0] % 2 == 0):
					im = concatenate((im[0:im.shape[0]/2,:-ext_fov_overlap/2], im_bottom), axis=1)									
				else:
					im = concatenate((im[0:im.shape[0]/2+1,:-ext_fov_overlap/2], im_bottom), axis=1)

	return im