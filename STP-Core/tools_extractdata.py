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
# Last modified: December, 11th 2016
#


import os
import os.path
import numpy
import time

from sys import argv, exit
from h5py import File as getHDF5
from numpy import float32

import stpio.tdf as tdf

def main(argv):    
	"""Extract a 2D image (projection or sinogram) from the input TDF file (DataExchange HDF5) and
	creates a 32-bit RAW file to disk.

	Parameters
	----------
	argv[0] : string
		The absolute path of the input TDF.

	argv[1] : int
		The relative position of the image within the dataset.

	argv[2] : string
		One of the following options: 'tomo', 'sino', 'flat', 'dark'.

	argv[3] : string
		The absolute path of the output 32-bit RAW image file. Filename will be modified by adding 
		image width, image height, minimum and maximum value of the input TDF dataset.

	Example
	-------
	tools_extractdata "S:\\dataset.tdf" 128 tomo "R:\\proj"	

	"""
	try:
		#
		# Get input parameters:
		#
		infile   = argv[0]
		index    = int(argv[1]) 
		imtype   = argv[2]
		outfile  = argv[3]		
	
		#
		# Body
		#	
	
		# Check if file exists:
		if not os.path.exists(infile):		
			#log = open(logfilename,"a")
			#log.write(os.linesep + "\tError: input TDF file not found. Process will end.")				
			#log.close()			
			exit()	

		# Open the HDF5 file:

		f = getHDF5( infile, 'r' )
		if (imtype == 'sino'):
			if "/tomo" in f:
				dset = f['tomo']	
			else: 
				dset = f['exchange/data']
			im = tdf.read_sino( dset, index )	
		elif (imtype == 'dark'):
			if "/dark" in f:
				dset = f['dark']	
			else: 
				dset = f['exchange/data_dark']	
			im = tdf.read_tomo( dset, index )
		elif (imtype == 'flat'):
			if "/flat" in f:
				dset = f['flat']	
			else: 
				dset = f['exchange/data_white']	
			im = tdf.read_tomo( dset, index )
		else:
			if "/tomo" in f:
				dset = f['tomo']	
			else: 
				dset = f['exchange/data']	
			im = tdf.read_tomo( dset, index )
				
		# Remove Infs e NaNs
		tmp = im[:].astype(numpy.float32)
		tmp = tmp[numpy.nonzero(numpy.isfinite(tmp))]	

		# Sort the gray levels:
		tmp = numpy.sort(tmp)
	
		# Return as minimum the value the skip 0.30% of "black" tail and 0.05% of "white" tail:
		low_idx  = int(tmp.shape[0] * 0.0030)
		high_idx = int(tmp.shape[0] * 0.9995)
		min = tmp[low_idx]
		max = tmp[high_idx]
		
		# Modify file name:
		outfile = outfile + '_' + str(im.shape[1]) + 'x' + str(im.shape[0]) + '_' + str(min) + '$' + str(max)	
		
		# Cast type:
		im = im.astype(float32)

		# Write RAW data to disk:
		im.tofile(outfile)			
	
	except:				
		
		exit()

if __name__ == "__main__":
	main(argv[1:])
