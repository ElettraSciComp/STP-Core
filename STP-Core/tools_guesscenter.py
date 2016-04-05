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

from math import pi
from numpy import float32, double, finfo, ndarray
from scipy.misc import imresize #scipy 0.12
from os import sep, remove
from os.path import basename
from sys import argv
from h5py import File as getHDF5

from pyfftw.interfaces.cache import enable as pyfftw_cache_enable, disable as pyfftw_cache_disable
from pyfftw.interfaces.cache import set_keepalive_time as pyfftw_set_keepalive_time

import time

import io.tdf as tdf
import utils.findcenter as findcenter
from utils.caching import cache2plan, plan2cache
from preprocess.extract_flatdark import extract_flatdark
from preprocess.flat_fielding import flat_fielding

from tifffile import imread, imsave # only for debug

def main(argv):          
	"""Try to guess the center of rotation of the input CT dataset.

    Parameters
    ----------
    infile  : array_like
        HDF5 input dataset

    outfile : string
        Full path where the identified center of rotation will be written as output

	scale   : int
        If sub-pixel precision is interesting, use e.g. 2.0 to get a center of rotation 
		of .5 value. Use 1.0 if sub-pixel precision is not required

	angles  : int
        Total number of angles of the input dataset	

	method : string
		One of the following options: "registration"

	tmppath : string
        Temporary path where look for cached flat/dark files
       
    """ 	   
	# Get path:
	infile  = argv[0]          # The HDF5 file on the
	outfile = argv[1]          # The txt file with the proposed center
	scale   = float(argv[2])
	angles  = float(argv[3])
	method  = argv[4]
	tmppath = argv[5]	
	if not tmppath.endswith(sep): tmppath += sep	

	pyfftw_cache_disable()
	pyfftw_cache_enable()
	pyfftw_set_keepalive_time(1800)	

	# Create a silly temporary log:
	tmplog  = tmppath + basename(infile) + str(time.time())
			
	# Open the HDF5 file (take into account also older TDF versions):
	f_in = getHDF5( infile, 'r' )
	if "/tomo" in f_in:
		dset = f_in['tomo']
	else: 
		dset = f_in['exchange/data']
	num_proj = tdf.get_nr_projs(dset)	
	num_sinos = tdf.get_nr_sinos(dset)	

	# Get flats and darks from cache or from file:
	try:
		corrplan = cache2plan(infile, tmppath)
	except Exception as e:
		#print "Error(s) when reading from cache"
		corrplan = extract_flatdark(f_in, True, tmplog)
		remove(tmplog)
		plan2cache(corrplan, infile, tmppath)

	# Get first and the 180 deg projections: 	
	im1 = tdf.read_tomo(dset,0).astype(float32)	

	idx = int(round(num_proj/angles * pi)) - 1
	im2 = tdf.read_tomo(dset,idx).astype(float32)		

	# Apply simple flat fielding (if applicable):
	if (isinstance(corrplan['im_flat_after'], ndarray) and isinstance(corrplan['im_flat'], ndarray) and
		isinstance(corrplan['im_dark'], ndarray) and isinstance(corrplan['im_dark_after'], ndarray)) :		
		im1 = ((abs(im1 - corrplan['im_dark'])) / (abs(corrplan['im_flat'] - corrplan['im_dark']) 
			+ finfo(float32).eps)).astype(float32)	
		im2 = ((abs(im2 - corrplan['im_dark_after'])) / (abs(corrplan['im_flat_after'] - corrplan['im_dark_after']) 
			+ finfo(float32).eps)).astype(float32)	

	# Scale projections (if required) to get subpixel estimation:
	if ( abs(scale - 1.0) > finfo(float32).eps ):	
		im1 = imresize(im1, (int(round(scale*im1.shape[0])), int(round(scale*im1.shape[1]))), interp='bicubic', mode='F');	
		im2 = imresize(im2, (int(round(scale*im2.shape[0])), int(round(scale*im2.shape[1]))), interp='bicubic', mode='F');	

	# Find the center (flipping left-right im2):
	cen = findcenter.usecorrelation(im1, im2[ :,::-1])
	cen = cen / scale
	
	# Print center to output file:
	text_file = open(outfile, "w")
	text_file.write(str(int(cen)))
	text_file.close()
	
	# Close input HDF5:
	f_in.close()

if __name__ == "__main__":
	main(argv[1:])
