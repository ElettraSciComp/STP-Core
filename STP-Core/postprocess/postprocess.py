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

from numpy import uint8, uint16

def postprocess( im, convert_opt, crop_opt ):	
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
