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

from os import remove, sep, linesep, listdir
from os.path import exists, dirname, basename, splitext

from numpy import array, finfo, copy, float32, reshape, fromfile, ndarray


def cache2plan(infile, cachepath):
	"""Read from cache the flat/dark images of the input TDF file.

    Parameters
    ----------    
    infile : string
		Absolute path of the input TDF dataset.
					
	Return value
	----------
	A structure with flat/dark images and related flags.
	
    """
	#path   = dirname(infile)
	path   = cachepath
	infile = splitext(basename(infile))[0]

	file_flat = [fn for fn in listdir(path) if fn.startswith(infile + '_imflat#')][0]
	file_flat_after = [fn for fn in listdir(path) if fn.startswith(infile + '_impostflat#')][0]

	skip_flat = (file_flat.split('$', 1)[1] == 'True')
	skip_flat_after = (file_flat_after.split('$', 1)[1] == 'True')
	
	im_flat = fromfile(path + file_flat, dtype=float32)	
	dim0 = int(file_flat.split('%', 1)[0].split('#',1)[-1])
	dim1 = int(file_flat.split('%', 1)[1].split('#',1)[0].split('$',1)[0])
	im_flat = reshape(im_flat, (dim1,dim0))

	im_flat_after = fromfile(path + file_flat_after, dtype=float32)	
	dim0 = int(file_flat_after.split('%', 1)[0].split('#',1)[-1])
	dim1 = int(file_flat_after.split('%', 1)[1].split('#',1)[0].split('$',1)[0])
	im_flat_after = reshape(im_flat_after, (dim1,dim0))

	file_dark = [fn for fn in listdir(path) if fn.startswith(infile + '_imdark#')][0]
	file_dark_after = [fn for fn in listdir(path) if fn.startswith(infile + '_impostdark#')][0]

	im_dark = fromfile(path + file_dark, dtype=float32)	
	dim0 = int(file_dark.split('%', 1)[0].split('#',1)[-1])
	dim1 = int(file_dark.split('%', 1)[1].split('#',1)[0])
	im_dark = reshape(im_dark, (dim1,dim0))

	im_dark_after = fromfile(path + file_dark_after, dtype=float32)	
	dim0 = int(file_dark_after.split('%', 1)[0].split('#',1)[-1])
	dim1 = int(file_dark_after.split('%', 1)[1].split('#',1)[0])
	im_dark_after = reshape(im_dark_after, (dim1,dim0))

	return {'im_flat':im_flat, 'im_flat_after':im_flat_after, 'im_dark':im_dark, 
			'im_dark_after':im_dark_after, 'skip_flat':skip_flat, 'skip_flat_after':skip_flat_after}



def plan2cache(corr_plan, infile, cachepath):
	"""Write to cache the flat/dark images of the input TDF file.

    Parameters
    ----------    
    infile : string
		Absolute path of the input TDF dataset.

	corr_plan : structure
		The plan with flat/dark images and flags.
					
	Return value
	----------
	No return value.
	
    """
	#path   = dirname(infile)
	path   = cachepath
	infile = splitext(basename(infile))[0]

	if (isinstance(corr_plan['im_flat'], ndarray)):
		im = corr_plan['im_flat'].astype(float32)
		outfile = path + infile + '_imflat#' + str(im.shape[1]) + '%' + str(im.shape[0]) + '$' + str(corr_plan['skip_flat'])
		im.tofile(outfile)

	if (isinstance(corr_plan['im_flat_after'], ndarray)):
		im = corr_plan['im_flat_after'].astype(float32)
		outfile = path + infile + '_impostflat#' + str(im.shape[1]) + '%' + str(im.shape[0]) + '$' + str(corr_plan['skip_flat_after'])
		im.tofile(outfile)

	if (isinstance(corr_plan['im_dark'], ndarray)):
		im = corr_plan['im_dark'].astype(float32)
		outfile = path + infile + '_imdark#' + str(im.shape[1]) + '%' + str(im.shape[0])
		im.tofile(outfile)

	if (isinstance(corr_plan['im_dark_after'], ndarray)):
		im = corr_plan['im_dark_after'].astype(float32)
		outfile = path + infile + '_impostdark#' + str(im.shape[1]) + '%' + str(im.shape[0])
		im.tofile(outfile)