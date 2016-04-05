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
 
import os
import os.path
import numpy
import time

from glob import glob
from sys import argv, exit
from h5py import File as getHDF5
from numpy import float32
from tifffile import imread, imsave

import io.tdf as tdf

def main(argv):    
	"""Extract a 2D image (projection or sinogram) from the input TDF file (DataExchange HDF5) and
	creates a 32-bit RAW file to disk.

    Parameters
    ----------
    argv[0] : string
		The absolute path of the input TDF.

	argv[1] : int
		The absolute path of output 32-bit TIFF image file.

	argv[2] : bool
		True to extract a projection, otherwise a sinogram is extracted.

	argv[3] : string
		The absolute path of the output 32-bit RAW image file. Filename will be modified by adding 
		image width, image height, minimum and maximum value of the input TDF dataset.

	Example
	-------
	tools_extractdata "S:\\dataset.tdf" 128 True "R:\\proj"	

    """
	try:
		   
		# Get input and output paths:
		inpath  = argv[0]
		outfile  = argv[1]  # The txt file with the proposed center
	
		if not inpath.endswith(os.path.sep): inpath += os.path.sep
	
		# Get the number of files in folder:
		files = sorted(glob(inpath + '*.tif*'))
		num_files = len(files)			
	
		# Read the median slice from disk:
		im = imread(files[num_files/2])
	
		# Flat the image and sort it:
		im_flat = im.flatten()
		im_flat = numpy.sort(im_flat)
	
		# Return as minimum the value the skip 0.30% of "black" tail and 0.005% of "white" tail:
		low_idx  = int(im_flat.shape[0] * 0.0030)
		high_idx = int(im_flat.shape[0] * 0.9995)
	
		min = im_flat[low_idx]
		max = im_flat[high_idx]
	
		# Print center to output file:
		text_file = open(outfile, "w")
		text_file.write( str(min) + ":" + str(max) )
		text_file.close()			
	
	except:				
		
		exit()

if __name__ == "__main__":
	main(argv[1:])
