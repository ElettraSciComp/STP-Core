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

from numpy import concatenate

from ringrem.rivers import rivers
from ringrem.boinhaibel import boinhaibel
from ringrem.munchetal import munchetal
from ringrem.raven import raven
from ringrem.miqueles import miqueles

def ring_correction (im, ringrem, flat_end, skip_flat_after, half_half, half_half_line, ext_fov):
	"""Apply ring artifacts compensation by de-striping the input sinogram.

    Parameters
    ----------
    im : array_like
		Image data (sinogram) as numpy array. 
	
	ringrem : string
		String containing ring removal method and parameters
	
	half_half : bool
		True to separately process the sinogram in two parts
	
	half_half_line : int
		Line number considered to identify the two parts to be processed separately. 
		(This parameter is ignored if half_half is False)
	
	skip_flat_after e ext_fov SERVE???
    
    """
	method, args = ringrem.split(":", 1)
			
	if (method == "rivers"):

		if flat_end and not skip_flat_after and half_half and not ext_fov:	
			im_top    = rivers( im[0:half_half_line,:], args)
			im_bottom = rivers( im[half_half_line:,:], args)
			im = concatenate((im_top,im_bottom), axis=0)					
		else:
			im = rivers(im, args)
					
	elif (method == "boinhaibel"):
		if flat_end and not skip_flat_after and half_half and not ext_fov:	
			im_top    = boinhaibel( im[0:half_half_line,:], args)
			im_bottom = boinhaibel( im[half_half_line:,:], args)
			im = concatenate((im_top,im_bottom), axis=0)
		else:
			im = boinhaibel(im, args)
					
	elif (method == "munchetal"):				
		im = munchetal(im, args)	
				
	elif (method == "raven"):				
		im = raven(im, args)

	return im 