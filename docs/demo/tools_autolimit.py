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
 
import os
import os.path
import numpy
import time

from glob import glob
from sys import argv, exit
from h5py import File as getHDF5
from numpy import float32
from tifffile import imread, imsave

# pystp-specific:
import stp_core.io.tdf as tdf

def main(argv):    
	"""Computes min/max limits to be used in image degradation to 8-bit or 16-bit.

    Parameters
    ----------
    argv[0] : string
		The absolute path of the input folder containing reconstructed TIFF files.

	argv[1] : string
		The absolute path of the output txt file with the proposed limits as string "min:max".

	Example
	-------
	tools_autolimit "S:\\SampleA\\slices" "R:\\Temp\\autolimit.txt"	

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
