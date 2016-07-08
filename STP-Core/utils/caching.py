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