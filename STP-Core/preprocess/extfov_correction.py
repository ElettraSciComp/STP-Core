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